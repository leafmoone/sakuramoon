"""Strict TOML configuration for bounded S000 engineering evidence."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import tomli_w
from pydantic import Field, StringConstraints, ValidationError, model_validator

from sakuramoon.config.schema import (
    AssetsConfig,
    CaptionConfig,
    CheckpointConfig,
    DistributedConfig,
    ExactFloat,
    FailureConfig,
    GradientConfig,
    GrowthConfig,
    KernelsConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    StrictModel,
    TimestepConfig,
    looks_like_unresolved_sentinel,
)


class EngineeringSmokeConfigurationError(ValueError):
    """A safe-to-render engineering-smoke configuration failure."""


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
RepositoryRelativePath = Annotated[
    str,
    StringConstraints(min_length=1, max_length=512),
]


class EngineeringEvidenceConfig(StrictModel):
    classification: Literal["synthetic_single_gpu_engineering_only"]
    formal_s000: Literal[False]
    production_cli_unlock: Literal[False]
    production_capacity_claim: Literal[False]
    production_quality_claim: Literal[False]


class EngineeringRunConfig(StrictModel):
    run_id: Annotated[
        str,
        StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$"),
    ]
    seed: NonNegativeInt
    optimizer_sr_seed: NonNegativeInt
    output_root: RepositoryRelativePath
    dependency_lock_path: Literal["uv.lock"]
    initial_successful_updates: Annotated[int, Field(ge=1, le=9)]
    fresh_process_next_updates: Literal[1]

    @property
    def total_successful_updates(self) -> int:
        return self.initial_successful_updates + self.fresh_process_next_updates

    @model_validator(mode="after")
    def validate_bound(self) -> EngineeringRunConfig:
        if not 1 <= self.total_successful_updates <= 10:
            raise ValueError("engineering smoke must remain within 1-10 updates")
        path = Path(self.output_root)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[:2] != ("artifacts", "engineering-smoke")
        ):
            raise ValueError(
                "engineering output_root must stay under artifacts/engineering-smoke"
            )
        return self


class EngineeringDeviceConfig(StrictModel):
    kind: Literal["cuda"]
    index: Literal[0]
    visible_device_count: Literal[1]
    world_size: Literal[1]


class EngineeringServiceConfig(StrictModel):
    socket_path: Literal["/run/sakuramoon/engineering-smoke-s000.sock"]
    ownership_lock_path: Literal["/run/sakuramoon/engineering-smoke-s000.lock"]
    request_timeout_seconds: Annotated[float, Field(gt=0.0, le=300.0)]
    startup_timeout_seconds: Annotated[float, Field(gt=0.0, le=300.0)]
    shutdown_timeout_seconds: Annotated[float, Field(gt=0.0, le=300.0)]
    fresh_process_timeout_seconds: Annotated[float, Field(gt=0.0, le=1800.0)]
    download_concurrency: Annotated[int, Field(ge=1, le=8)]
    verified_shard_lookahead: Annotated[int, Field(ge=1, le=8)]
    lease_channel_capacity: Annotated[int, Field(ge=1, le=8)]
    ack_channel_capacity: Annotated[int, Field(ge=1, le=8)]


class EngineeringDataConfig(StrictModel):
    source_kind: Literal["generated_local_tar"]
    source_width: Literal[640]
    source_height: Literal[512]
    shard_count: Annotated[int, Field(ge=4, le=16)]
    samples_per_shard: Literal[1]
    validation_sample_count: Literal[2000]
    persistent_workers: Literal[2]
    ready_batches: Literal[2]
    local_batch: Literal[1]
    pass_index: Literal[0]
    pin_memory: Literal[True]
    drop_last: Literal[True]
    cache_low_bytes: NonNegativeInt
    cache_high_bytes: PositiveInt
    transport_chunk_bytes: PositiveInt
    base_area_px: Literal[262144]
    quantum_px: Literal[32]
    min_short_edge_px: Literal[256]
    max_aspect_ratio: Annotated[ExactFloat, Field(ge=4.0, le=4.0)]
    shape_count: Literal[17]
    transpose_closed: Literal[True]
    exif_transpose: Literal[True]
    color_mode: Literal["RGB"]
    no_upscale: Literal[True]
    preserve_aspect_ratio: Literal[True]
    allow_padding: Literal[False]
    min_crop_retention: Annotated[ExactFloat, Field(ge=0.8, le=0.8)]

    @model_validator(mode="after")
    def validate_data_bounds(self) -> EngineeringDataConfig:
        if self.cache_low_bytes >= self.cache_high_bytes:
            raise ValueError("engineering cache low watermark must be below high")
        if self.ready_batches != self.persistent_workers:
            raise ValueError("engineering ready batch count must equal worker count")
        return self


class EngineeringStageConfig(StrictModel):
    name: Literal["S0"]
    depth: Literal[16]
    resolution: Literal[256]
    local_batch: Literal[1]
    accumulation: Literal[1]
    successful_updates: Annotated[int, Field(ge=1, le=10)]
    activation_checkpoint_mode: Literal["none"]
    growth_alpha: Annotated[ExactFloat, Field(ge=1.0, le=1.0)]


class EngineeringMeasurementConfig(StrictModel):
    noise_observation_boundary: Annotated[
        ExactFloat,
        Field(ge=0.95, le=0.95),
    ]


class EngineeringSmokeConfig(StrictModel):
    schema_version: Literal[1]
    evidence: EngineeringEvidenceConfig
    run: EngineeringRunConfig
    device: EngineeringDeviceConfig
    service: EngineeringServiceConfig
    assets: AssetsConfig
    data: EngineeringDataConfig
    caption: CaptionConfig
    model: ModelConfig
    objective: ObjectiveConfig
    timestep: TimestepConfig
    optimizer: OptimizerConfig
    gradient: GradientConfig
    distributed: DistributedConfig
    checkpoint: CheckpointConfig
    growth: GrowthConfig
    stage: EngineeringStageConfig
    measurement: EngineeringMeasurementConfig
    kernels: KernelsConfig
    failure: FailureConfig

    @model_validator(mode="after")
    def validate_engineering_boundary(self) -> EngineeringSmokeConfig:
        if self.stage.successful_updates != self.run.total_successful_updates:
            raise ValueError("stage update bound differs from run update bound")
        if self.stage.local_batch != self.data.local_batch:
            raise ValueError("stage and data local batch differ")
        if (
            self.distributed.backend != "native"
            or self.distributed.world_size != 1
            or self.kernels.attention_backend != "dense_sdpa_reference"
            or self.kernels.silent_fallback
        ):
            raise ValueError("engineering smoke requires native single-GPU dense SDPA")
        if (
            self.growth.enabled
            or self.stage.growth_alpha != 1.0
            or self.checkpoint.kind != "raw"
        ):
            raise ValueError("engineering smoke requires fixed-depth raw checkpointing")
        if (
            self.model.dit.hidden_size != 2560
            or self.model.dit.q_heads != 20
            or self.model.dit.kv_heads != 5
            or self.assets.vae.latent_channels != 128
        ):
            raise ValueError("engineering model differs from the locked 16L S0 shape")
        return self


@dataclass(frozen=True, slots=True)
class LoadedEngineeringSmokeConfig:
    config: EngineeringSmokeConfig
    input_path: str
    input_sha256: str
    resolved_toml: str
    resolved_sha256: str


def _contains_sentinel(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_sentinel(item) for item in cast(dict[object, object], value).values())
    if isinstance(value, list):
        return any(_contains_sentinel(item) for item in cast(list[object], value))
    return looks_like_unresolved_sentinel(value)


def _safe_error(error: ValidationError) -> EngineeringSmokeConfigurationError:
    locations = sorted(
        ".".join(str(part) for part in item["loc"]) or "<root>"
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    )
    return EngineeringSmokeConfigurationError(
        "engineering-smoke configuration validation failed at: "
        + ", ".join(locations)
    )


def _resolve_input(config_path: Path, config_root: Path) -> tuple[Path, Path]:
    try:
        root = config_root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise OSError
        requested = config_path if config_path.is_absolute() else root / config_path
        relative = requested.relative_to(root)
        if ".." in relative.parts or requested.is_symlink():
            raise OSError
        path = requested.resolve(strict=True)
        path.relative_to(root)
        if path.suffix != ".toml" or not path.is_file():
            raise OSError
    except (OSError, ValueError):
        raise EngineeringSmokeConfigurationError(
            "engineering-smoke config must be a real TOML file inside config_root"
        ) from None
    return root, path


def load_engineering_smoke_config(
    config_path: Path,
    *,
    config_root: Path,
) -> LoadedEngineeringSmokeConfig:
    """Load one standalone strict TOML document without production sentinels."""

    root, path = _resolve_input(config_path, config_root)
    try:
        raw = path.read_bytes()
        payload = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise EngineeringSmokeConfigurationError(
            "engineering-smoke config is unreadable or invalid TOML"
        ) from None
    if _contains_sentinel(payload):
        raise EngineeringSmokeConfigurationError(
            "engineering-smoke config contains an unresolved sentinel"
        )
    try:
        config = EngineeringSmokeConfig.model_validate(payload)
    except ValidationError as error:
        raise _safe_error(error) from None
    resolved = tomli_w.dumps(config.model_dump(mode="python")).encode("utf-8")
    return LoadedEngineeringSmokeConfig(
        config=config,
        input_path=path.relative_to(root).as_posix(),
        input_sha256=hashlib.sha256(raw).hexdigest(),
        resolved_toml=resolved.decode("utf-8"),
        resolved_sha256=hashlib.sha256(resolved).hexdigest(),
    )


def require_single_gpu_environment(config: EngineeringSmokeConfig) -> None:
    """Reject unavailable CUDA, extra visible devices, or distributed launch state."""

    import os

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EngineeringSmokeConfigurationError(
            "engineering smoke requires exactly one visible CUDA device"
        )
    if torch.cuda.current_device() != config.device.index:
        raise EngineeringSmokeConfigurationError(
            "current CUDA device differs from the TOML selection"
        )
    distributed_names = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    if any(name in os.environ for name in distributed_names):
        raise EngineeringSmokeConfigurationError(
            "engineering smoke refuses a distributed launch environment"
        )


def is_sha256(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


__all__ = [
    "EngineeringSmokeConfig",
    "EngineeringSmokeConfigurationError",
    "LoadedEngineeringSmokeConfig",
    "is_sha256",
    "load_engineering_smoke_config",
    "require_single_gpu_environment",
]
