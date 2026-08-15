"""Pydantic schema for all runtime configuration surfaces."""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

from sakuramoon.sampling.profiles import (
    SAMPLING_PROFILES,
    SamplingProfile,
    SamplingProfileName,
    SamplingSolver,
    TimeSchedule,
    resolve_sampling_profile,
)


def _toml_array_to_tuple(value: object) -> object:
    if type(value) is list:
        return tuple(cast(list[object], value))
    return value


def _require_toml_float(value: object) -> object:
    if type(value) is not float:
        raise ValueError("value must use TOML float syntax")
    return value


StringTuple = Annotated[tuple[str, ...], BeforeValidator(_toml_array_to_tuple)]
IntTuple = Annotated[tuple[int, ...], BeforeValidator(_toml_array_to_tuple)]
ExactFloat = Annotated[float, BeforeValidator(_require_toml_float)]
PositiveFloat = Annotated[ExactFloat, Field(gt=0.0)]
NonNegativeFloat = Annotated[ExactFloat, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
BoundedQueueCapacity = Annotated[int, Field(ge=1, le=1024)]
TelemetryEventTimeout = Annotated[ExactFloat, Field(gt=0.0, le=300.0)]
FIXED_TIMING_PHASES = (
    "data",
    "qwen",
    "vae",
    "conditioning",
    "dit_forward",
    "loss",
    "backward",
    "ddp",
    "clip",
    "optimizer",
    "checkpoint",
    "evaluation",
    "cache",
    "tar",
    "json",
    "caption",
    "tokenize",
    "decode",
    "exif",
    "crop",
    "bucket",
    "h2d",
    "condition",
    "zero_grad",
    "sample",
)
SecretEnvName = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$", min_length=2, max_length=128),
]
FixedZero = Annotated[ExactFloat, Field(ge=0.0, le=0.0)]
FixedOne = Annotated[ExactFloat, Field(ge=1.0, le=1.0)]
FixedPointOne = Annotated[ExactFloat, Field(ge=0.1, le=0.1)]
FixedPointTwo = Annotated[ExactFloat, Field(ge=0.2, le=0.2)]
FixedPointThree = Annotated[ExactFloat, Field(ge=0.3, le=0.3)]
FixedPointEight = Annotated[ExactFloat, Field(ge=0.8, le=0.8)]
FixedPointNineFive = Annotated[ExactFloat, Field(ge=0.95, le=0.95)]
FixedFour = Annotated[ExactFloat, Field(ge=4.0, le=4.0)]
FixedSixteen = Annotated[ExactFloat, Field(ge=16.0, le=16.0)]
FixedThousand = Annotated[ExactFloat, Field(ge=1000.0, le=1000.0)]
FixedNormEps = Annotated[ExactFloat, Field(ge=0.000001, le=0.000001)]
FixedNegativePointEight = Annotated[ExactFloat, Field(ge=-0.8, le=-0.8)]
FixedPointZeroFive = Annotated[ExactFloat, Field(ge=0.05, le=0.05)]
FixedTwoPointNine = Annotated[ExactFloat, Field(ge=2.9, le=2.9)]
FixedOptimizerEps = Annotated[ExactFloat, Field(ge=0.00000001, le=0.00000001)]
WeightDecay = Annotated[ExactFloat, Field(ge=0.0, le=1.0)]
FixedPointZeroTwo = Annotated[ExactFloat, Field(ge=0.02, le=0.02)]


class StrictModel(BaseModel):
    """Base for immutable, exact-type, unknown-key rejecting config tables."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class RunConfig(StrictModel):
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    intent: Literal["train", "eval", "sample", "template"]
    stage: Literal["S0", "S1", "G1", "S2", "G2", "S3", "H1", "H2"]
    seed: NonNegativeInt


class PathsConfig(StrictModel):
    run_dir: Annotated[str, StringConstraints(min_length=1)]
    cache_dir: Annotated[str, StringConstraints(min_length=1)]
    checkpoint_dir: Annotated[str, StringConstraints(min_length=1)]
    artifact_dir: Annotated[str, StringConstraints(min_length=1)]


class StorageConfig(StrictModel):
    mode: Literal["server_backed"]
    shared_filesystem: Literal["nfs"]
    shared_mount_source: Annotated[
        str, StringConstraints(min_length=1, max_length=1024)
    ]
    nfs_version: Literal[3]
    hard_mount: Literal[True]
    minimum_free_gib: NonNegativeInt
    measured_raw_checkpoint_bytes: PositiveInt
    checkpoint_copies: PositiveInt
    atomic_publish_probe: Literal[True]


class SecurityConfig(StrictModel):
    modelscope_token_env: SecretEnvName
    wandb_api_key_env: SecretEnvName

    @model_validator(mode="after")
    def reject_secret_shaped_names(self) -> SecurityConfig:
        for field_name, value in (
            ("modelscope_token_env", self.modelscope_token_env),
            ("wandb_api_key_env", self.wandb_api_key_env),
        ):
            if value.endswith(("_VALUE", "_SECRET")):
                raise ValueError(f"{field_name} must name an environment variable")
        return self


class QwenAssetConfig(StrictModel):
    local_path: Literal["model/qwen_3.5_2B"]
    dtype: Literal["bfloat16"]
    frozen: Literal[True]
    layers: Literal[24]
    hidden_size: Literal[2048]
    use_cache: Literal[False]
    visual_path_enabled: Literal[False]


class VaeAssetConfig(StrictModel):
    local_path: Literal["model/vae"]
    dtype: Literal["bfloat16"]
    frozen: Literal[True]
    latent_channels: Literal[128]
    downsample_factor: Literal[16]
    sample_posterior: Literal[False]


class AssetsConfig(StrictModel):
    qwen: QwenAssetConfig
    vae: VaeAssetConfig


class DataSourceConfig(StrictModel):
    repo_id: Literal["leafmoone/webdataset_danbooru_v2"]
    revision: Literal["master"]


class DataManifestConfig(StrictModel):
    path: Annotated[str, StringConstraints(min_length=1)]
    initialize_if_missing: Literal[True]
    refresh_existing: Literal[False]

    @model_validator(mode="after")
    def validate_path(self) -> DataManifestConfig:
        path = PurePosixPath(self.path)
        if (
            self.path != self.path.strip()
            or "\\" in self.path
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.path
            or not path.name
        ):
            raise ValueError(
                "data manifest path must be a normalized repository-relative file"
            )
        return self


class DataCacheConfig(StrictModel):
    low_watermark_gib: Annotated[int, Field(ge=0, lt=500)]
    high_watermark_gib: Annotated[int, Field(gt=0, le=500)]
    download_concurrency: PositiveInt
    verified_shard_lookahead: PositiveInt
    persistent_workers_per_rank: PositiveInt
    ready_batches_per_rank: PositiveInt

    @model_validator(mode="after")
    def validate_watermarks(self) -> DataCacheConfig:
        if self.low_watermark_gib >= self.high_watermark_gib:
            raise ValueError("cache low watermark must be below high watermark")
        return self


class DataServiceConfig(StrictModel):
    socket_path: Literal["/run/sakuramoon/data-service.sock"]
    ownership_lock_path: Literal["/run/sakuramoon/data-service.lock"]
    mainset_path: Annotated[str, StringConstraints(min_length=1)]
    request_timeout_seconds: Annotated[ExactFloat, Field(gt=0.0, le=300.0)]
    lease_channel_capacity: PositiveInt
    ack_channel_capacity: PositiveInt


class DataTransportConfig(StrictModel):
    connect_timeout_seconds: Annotated[ExactFloat, Field(gt=0.0, le=300.0)]
    read_timeout_seconds: Annotated[ExactFloat, Field(gt=0.0, le=300.0)]
    max_retries: Annotated[int, Field(ge=0, le=10)]
    retry_backoff_seconds: Annotated[ExactFloat, Field(ge=0.0, le=60.0)]
    stream_chunk_bytes: Annotated[int, Field(ge=65536, le=16777216)]
    streams_per_shard: Annotated[int, Field(ge=1, le=16)]


class DataLoaderConfig(StrictModel):
    pin_memory: bool
    drop_last: bool
    length_sort_window_batches: Annotated[int, Field(ge=1, le=8)] = 1


class DataValidationConfig(StrictModel):
    selection_path: Annotated[str, StringConstraints(min_length=1)]
    shard_root: Annotated[str, StringConstraints(min_length=1)]
    shard_count: PositiveInt

    @model_validator(mode="after")
    def validate_paths(self) -> DataValidationConfig:
        for name, value in (
            ("selection_path", self.selection_path),
            ("shard_root", self.shard_root),
        ):
            path = PurePosixPath(value)
            if (
                value != value.strip()
                or "\\" in value
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != value
                or not path.name
            ):
                raise ValueError(
                    f"data validation {name} must be a normalized repository-relative path"
                )
        if self.selection_path == self.shard_root:
            raise ValueError("data validation selection and shard paths must differ")
        return self


class DataImageConfig(StrictModel):
    exif_transpose: Literal[True]
    color_mode: Literal["RGB"]
    no_upscale: Literal[True]
    preserve_aspect_ratio: Literal[True]
    allow_padding: Literal[False]
    min_crop_retention: FixedPointEight


class DataBucketsConfig(StrictModel):
    base_area_px: Literal[262144]
    quantum_px: Literal[32]
    min_short_edge_px: Literal[256]
    max_aspect_ratio: FixedFour
    shape_count: Literal[17]
    transpose_closed: Literal[True]


class DataConfig(StrictModel):
    source: DataSourceConfig
    manifest: DataManifestConfig
    cache: DataCacheConfig
    service: DataServiceConfig
    transport: DataTransportConfig
    loader: DataLoaderConfig
    validation: DataValidationConfig
    image: DataImageConfig
    buckets: DataBucketsConfig

    @model_validator(mode="after")
    def validate_service_worker_capacities(self) -> DataConfig:
        workers = self.cache.persistent_workers_per_rank
        if (
            self.cache.verified_shard_lookahead < workers
            or self.service.lease_channel_capacity < workers
            or self.service.ack_channel_capacity < workers
            or self.cache.ready_batches_per_rank < workers
            or self.cache.ready_batches_per_rank % workers
        ):
            raise ValueError(
                "data service and ready channel capacities must cover exact worker topology"
            )
        return self


class NlDropoutConfig(StrictModel):
    long_names: FixedPointThree
    long_no_names: FixedPointThree
    short_vibes: FixedPointThree
    nl2: FixedPointThree
    nl3: FixedPointThree

    @model_validator(mode="after")
    def require_equal_probabilities(self) -> NlDropoutConfig:
        values = {
            self.long_names,
            self.long_no_names,
            self.short_vibes,
            self.nl2,
            self.nl3,
        }
        if len(values) != 1:
            raise ValueError("all five NL dropout probabilities must be equal")
        return self


class CaptionDropoutConfig(StrictModel):
    all_condition: FixedPointOne
    tag: FixedPointOne
    candidate_source: FixedPointThree
    nl: NlDropoutConfig


class CaptionConfig(StrictModel):
    category_order: StringTuple
    tag_separator: Literal[", "]
    tag_nl_separator: Literal["\n\n"]
    text_condition_max: Literal[512]
    condition_buckets: IntTuple
    qwen_dense_lengths: IntTuple
    dropout: CaptionDropoutConfig

    @model_validator(mode="after")
    def validate_protocol_sequences(self) -> CaptionConfig:
        if self.category_order != (
            "tags",
            "nl",
            "style_condition",
        ):
            raise ValueError(
                "caption category order differs from the approved protocol"
            )
        if self.condition_buckets != (64, 128, 192, 256, 320, 384, 448, 512):
            raise ValueError("condition buckets differ from the approved eight buckets")
        if self.qwen_dense_lengths != (98, 162, 226, 290, 354, 418, 482, 546):
            raise ValueError("Qwen dense lengths differ from the approved protocol")
        return self


class TextModelConfig(StrictModel):
    hidden_state_blocks: IntTuple
    input_size: Literal[2048]
    adapter_size: Literal[1024]
    output_size: Literal[2560]
    groups: Literal[8]
    attention_heads: Literal[16]
    bidirectional_attention_layers: Literal[1]
    no_positional_encoding: Literal[True]
    norm_eps: FixedNormEps
    mix_gate_init: FixedZero
    layer_scale_init: FixedOne
    projection_bias: Literal[False]
    linear_dtype: Literal["bfloat16"]
    sensitive_dtype: Literal["float32"]

    @model_validator(mode="after")
    def validate_blocks(self) -> TextModelConfig:
        if self.hidden_state_blocks != (2, 4, 8, 12, 16, 20, 24):
            raise ValueError(
                "text hidden-state blocks differ from the approved mapping"
            )
        return self


class StyleModelConfig(StrictModel):
    query_count: Literal[4]
    input_size: Literal[2048]
    hidden_size: Literal[1024]
    mlp_intermediate_size: Literal[2048]
    output_size: Literal[2560]
    null_tokens_learned: Literal[True]
    attention_heads: Literal[16]
    norm_eps: FixedNormEps
    init_std: FixedPointZeroTwo
    projection_bias: Literal[False]
    linear_dtype: Literal["bfloat16"]
    sensitive_dtype: Literal["float32"]


class PackingModelConfig(StrictModel):
    order: StringTuple
    style_tokens: Literal[4]
    remove_text_padding: Literal[True]
    cross_sample_attention: Literal[False]
    modality_init_std: FixedPointZeroTwo

    @model_validator(mode="after")
    def validate_order(self) -> PackingModelConfig:
        if self.order != ("text", "style", "image"):
            raise ValueError("packing order must be text, style, image")
        return self


class RopeModelConfig(StrictModel):
    head_dim: Literal[128]
    nope_dim: Literal[32]
    y_dim: Literal[48]
    x_dim: Literal[48]
    position_scale: FixedSixteen
    theta: FixedThousand
    cell_center: Literal[True]
    area_normalized: Literal[True]


class DitModelConfig(StrictModel):
    hidden_size: Literal[2560]
    head_dim: Literal[128]
    q_heads: Literal[20]
    kv_heads: Literal[5]
    intermediate_size: Literal[6912]
    stable_slot_count: Literal[24]
    patch_size: Literal[1]
    attention_dropout: FixedZero
    mlp_dropout: FixedZero
    projection_bias: Literal[False]
    norm_eps: FixedNormEps
    norm_accumulation: Literal["float32"]
    activation_dtype: Literal["bfloat16"]


class ConditionModelConfig(StrictModel):
    timestep_dim: Literal[256]
    size_dim: Literal[64]
    aspect_dim: Literal[64]
    input_dim: Literal[384]
    hidden_dim: Literal[1024]
    block_modulation_chunks: Literal[6]
    shared_projection_zero_init: Literal[True]
    per_block_bias_zero_init: Literal[True]


class HeadModelConfig(StrictModel):
    final_modulation_size: Literal[5120]
    out_channels: Literal[128]
    prediction_type: Literal["x"]
    image_span_only: Literal[True]
    weight_zero_init: Literal[True]
    bias_zero_init: Literal[True]


class ModelConfig(StrictModel):
    text: TextModelConfig
    style: StyleModelConfig
    packing: PackingModelConfig
    rope: RopeModelConfig
    dit: DitModelConfig
    condition: ConditionModelConfig
    head: HeadModelConfig


class ObjectiveConfig(StrictModel):
    prediction_type: Literal["x"]
    loss: Literal["jlt_x_prediction_velocity_mse"]
    target_velocity: Literal["x_to_v(clean,state,t,t_eps)"]
    endpoint_weighting: Literal["inverse_square_clamped"]
    interpolation: Literal["z_t=t*x+(1-t)*epsilon"]
    velocity_loss_dtype: Literal["float32"]
    reduction: Literal["per_sample_then_global_sample_mean"]


class TimestepConfig(StrictModel):
    distribution: Literal["jlt"]
    p_mean: FixedNegativePointEight
    p_std: FixedPointEight
    noise_scale: FixedOne
    t_eps: FixedPointZeroFive


class SamplingProfileConfig(StrictModel):
    solver: SamplingSolver
    steps: PositiveInt
    time_schedule: TimeSchedule


class SamplingProfilesConfig(StrictModel):
    preview: SamplingProfileConfig
    balanced: SamplingProfileConfig
    reference: SamplingProfileConfig

    @model_validator(mode="after")
    def validate_canonical_registry(self) -> SamplingProfilesConfig:
        for name in ("preview", "balanced", "reference"):
            configured = getattr(self, name)
            canonical = SAMPLING_PROFILES[name]
            if (
                configured.solver != canonical.solver
                or configured.steps != canonical.steps
                or configured.time_schedule != canonical.time_schedule
            ):
                raise ValueError(
                    f"sampling profile {name} differs from the approved registry"
                )
        return self


class TrainingSamplingConfig(StrictModel):
    """Periodic image samples made from captions seen by the training loop."""

    enabled: bool = True
    every_updates: PositiveInt = 1000
    image_count: PositiveInt = 12
    output_subdir: Annotated[str, StringConstraints(min_length=1)] = "sample"

    @model_validator(mode="after")
    def validate_output_subdir(self) -> TrainingSamplingConfig:
        path = PurePosixPath(self.output_subdir)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or any(part in {"", "."} for part in path.parts)
        ):
            raise ValueError(
                "training sample output_subdir must be repository-relative"
            )
        return self


class SamplingConfig(StrictModel):
    profile: SamplingProfileName
    profiles: SamplingProfilesConfig
    state_dtype: Literal["float32"]
    training: TrainingSamplingConfig = TrainingSamplingConfig()

    @property
    def selected(self) -> SamplingProfile:
        return resolve_sampling_profile(self.profile)

    @computed_field
    @property
    def solver(self) -> SamplingSolver:
        return self.selected.solver

    @computed_field
    @property
    def steps(self) -> int:
        return self.selected.steps

    @computed_field
    @property
    def nfe(self) -> int:
        return self.selected.nfe

    @computed_field
    @property
    def time_schedule(self) -> TimeSchedule:
        return self.selected.time_schedule


class CfgConfig(StrictModel):
    scale: FixedTwoPointNine
    full_interval: Literal[True]
    rescale: Literal[False]
    conversion_order: Literal["each_x_to_v_then_cfg"]


class OptimizerConfig(StrictModel):
    name: Literal["torchao_adamw8bit"]
    base_lr: PositiveFloat
    reference_batch: PositiveInt
    lr_scaling: Literal["linear_global_batch"]
    betas: Annotated[
        tuple[ExactFloat, ExactFloat], BeforeValidator(_toml_array_to_tuple)
    ]
    eps: FixedOptimizerEps
    block_size: Literal[256]
    bf16_stochastic_round: Literal[True]
    matrix_weight_decay: WeightDecay
    sensitive_weight_decay: WeightDecay

    @model_validator(mode="after")
    def validate_betas(self) -> OptimizerConfig:
        if self.betas != (0.9, 0.95):
            raise ValueError("optimizer betas differ from the approved values")
        return self


class SchedulerConfig(StrictModel):
    name: Literal["linear_warmup_constant"]
    warmup_updates: PositiveInt
    after_warmup: Literal["constant"]


class GradientConfig(StrictModel):
    clip_norm: FixedOne
    clip_dtype: Literal["float32"]
    global_sample_mean: Literal[True]


class DistributedConfig(StrictModel):
    backend: Literal["native", "accelerate", "ddp"]
    world_size: Annotated[int, Field(ge=1, le=4)]
    frozen_encoders_outside_wrapper: Literal[True]
    automatic_backend_fallback: Literal[False]


class CheckpointConfig(StrictModel):
    kind: Literal["raw"]
    full_every_updates: PositiveInt
    slots: PositiveInt
    atomic_complete_marker: Literal[True]
    canonical_fqn: Literal[True]


class GrowthConfig(StrictModel):
    enabled: bool
    alpha_schedule: Literal["half_cosine"]
    alpha_fraction: FixedPointZeroTwo
    min_updates: Literal[1000]
    max_updates: Literal[5000]
    random_new_slots: Literal[True]
    copy_old_slots: Literal[False]


StageName = Literal["S0", "S1", "G1", "S2", "G2", "S3", "H1", "H2"]


class StageConfig(StrictModel):
    name: StageName
    enabled: bool
    predecessor: Literal["", "S0", "S1", "G1", "S2", "G2", "S3", "H1"]
    world_size: Annotated[int, Field(ge=1, le=4)]
    depth: Literal[16, 20, 24]
    resolution: Literal[256, 512, 768, 1024]
    local_batch: PositiveInt
    accumulation: PositiveInt
    global_batch: PositiveInt
    activation_checkpoint_mode: Literal["none", "alternating", "all"]
    planned_updates: PositiveInt
    manual_finalize: Literal[True]
    automatic_transition: Literal[False]

    @model_validator(mode="after")
    def validate_transition(self) -> StageConfig:
        expected_global_batch = self.local_batch * self.accumulation * self.world_size
        if self.global_batch != expected_global_batch:
            raise ValueError(
                "stage global_batch must equal local_batch * accumulation * world_size"
            )
        return self


class KernelsConfig(StrictModel):
    attention_backend: Literal["dense_sdpa_reference", "das_fa2_varlen"]
    qwen_attention_backend: Literal["sdpa", "flash_attention_2"] = "sdpa"
    tunableop_enabled: bool = False
    tunableop_tuning: bool = False
    tunableop_record_untuned: bool = False
    tunableop_max_tuning_duration_ms: Annotated[int, Field(ge=1, le=1000)] = 50
    torch_compile_enabled: bool = False
    torch_compile_backend: Literal["inductor"] = "inductor"
    torch_compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune-no-cudagraphs"
    ] = "default"
    torch_compile_dynamic: bool = False
    dtype: Literal["bfloat16"]
    native_gqa: Literal[True]
    repeat_kv_heads: Literal[False]
    silent_fallback: Literal[False]


class FailureConfig(StrictModel):
    stop_on_nonfinite: Literal[True]
    stop_on_oom: Literal[True]
    stop_on_rank_divergence: Literal[True]
    allow_force_bypass: Literal[False]
    automatic_batch_change: Literal[False]
    automatic_backend_change: Literal[False]


class LoggingConfig(StrictModel):
    local_jsonl_path: Annotated[str, StringConstraints(min_length=1)]
    flush_every_updates: PositiveInt
    async_remote: Literal[True]
    noise_observation_boundary: FixedPointNineFive
    observer_queue_capacity: BoundedQueueCapacity
    observer_event_timeout_seconds: TelemetryEventTimeout


class WandbConfig(StrictModel):
    enabled: bool
    project: Annotated[str, StringConstraints(min_length=1)]
    entity: Annotated[str, StringConstraints(min_length=1)]
    offline_on_network_error: Literal[True]
    retry_jsonl_path: Annotated[str, StringConstraints(min_length=1)]
    queue_capacity: BoundedQueueCapacity
    replay_retry_on_start: Literal[True]
    finish_on_close: Literal[True]
    resume_policy: Literal["allow"]


class TimingConfig(StrictModel):
    enabled: bool
    cuda_events: Literal[True]
    force_synchronize_each_phase: Literal[False]
    phases: StringTuple

    @model_validator(mode="after")
    def validate_phases(self) -> TimingConfig:
        if self.phases != FIXED_TIMING_PHASES:
            raise ValueError(
                "timing phases must exactly match the fixed ordered vocabulary"
            )
        return self


class EvaluationDisabledConfig(StrictModel):
    enabled: Literal[False]


class EvaluationEnabledConfig(StrictModel):
    enabled: Literal[True]
    every_updates: PositiveInt = 1000
    sample_count: PositiveInt
    real_sample_count: PositiveInt
    batch_size: PositiveInt
    is_splits: PositiveInt
    kid_subsets: PositiveInt = 100
    kid_subset_size: PositiveInt = 100
    prompt_path: Annotated[str, StringConstraints(min_length=1)]
    validation_shard_root: Annotated[str, StringConstraints(min_length=1)]
    output_dir: Annotated[str, StringConstraints(min_length=1)]
    sampling_profile: Literal["preview", "balanced", "reference"]

    @model_validator(mode="before")
    @classmethod
    def default_real_sample_count(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        payload = cast(dict[str, object], value)
        if "real_sample_count" in payload or "sample_count" not in payload:
            return value
        return {
            **payload,
            "real_sample_count": payload["sample_count"],
        }

    @model_validator(mode="after")
    def validate_evaluation(self) -> EvaluationEnabledConfig:
        if self.sample_count < 2 or self.sample_count % self.is_splits:
            raise ValueError(
                "evaluation sample_count must divide evenly into IS splits"
            )
        if self.kid_subset_size < 2 or self.kid_subset_size > min(
            self.sample_count, self.resolved_real_sample_count
        ):
            raise ValueError("evaluation KID subset size exceeds the sample count")
        for value in (self.prompt_path, self.validation_shard_root, self.output_dir):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("evaluation paths must be repository-relative")
        return self

    @property
    def resolved_real_sample_count(self) -> int:
        return self.real_sample_count


EvaluationConfig = Annotated[
    EvaluationDisabledConfig | EvaluationEnabledConfig,
    Field(discriminator="enabled"),
]


class RuntimeConfig(StrictModel):
    schema_version: Literal[1]
    run: RunConfig
    paths: PathsConfig
    storage: StorageConfig
    security: SecurityConfig
    assets: AssetsConfig
    data: DataConfig
    caption: CaptionConfig
    model: ModelConfig
    objective: ObjectiveConfig
    timestep: TimestepConfig
    sampling: SamplingConfig
    cfg: CfgConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    gradient: GradientConfig
    distributed: DistributedConfig
    checkpoint: CheckpointConfig
    growth: GrowthConfig
    stage: StageConfig
    kernels: KernelsConfig
    failure: FailureConfig
    logging: LoggingConfig
    wandb: WandbConfig
    timing: TimingConfig
    evaluation: EvaluationConfig

    def scaled_learning_rate(self) -> float:
        """Return the JLT-style LR for the configured effective global batch."""

        learning_rate = (
            self.optimizer.base_lr
            * self.stage.global_batch
            / self.optimizer.reference_batch
        )
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "scaled optimizer learning rate must be finite and positive"
            )
        return learning_rate

    @model_validator(mode="after")
    def validate_cross_table_contract(self) -> RuntimeConfig:
        self.scaled_learning_rate()
        if self.run.stage != self.stage.name:
            raise ValueError("run.stage and stage.name must match")
        if self.run.intent == "template":
            if self.stage.name not in {"H1", "H2"} or self.stage.enabled:
                raise ValueError(
                    "template intent is reserved for disabled H1/H2 configurations"
                )
        elif not self.stage.enabled:
            raise ValueError("selected stage must be enabled")
        accepted_backends = (
            {"native", "accelerate"} if self.stage.name == "S0" else {"ddp"}
        )
        if self.distributed.backend not in accepted_backends:
            raise ValueError("distributed backend does not match the selected stage")
        if self.stage.name == "S0" and (
            (self.distributed.backend == "native" and self.distributed.world_size != 1)
            or (
                self.distributed.backend == "accelerate"
                and self.distributed.world_size <= 1
            )
        ):
            raise ValueError("S0 distributed backend and world size are inconsistent")
        if self.distributed.world_size != self.stage.world_size:
            raise ValueError("distributed and stage world_size must match")
        if self.growth.enabled != (self.stage.name in {"G1", "G2"}):
            raise ValueError("growth is enabled only for G1 and G2")
        artifact_root = PurePosixPath(self.paths.artifact_dir)
        metric_path = PurePosixPath(self.logging.local_jsonl_path)
        retry_path = PurePosixPath(self.wandb.retry_jsonl_path)
        for label, path in (
            ("logging.local_jsonl_path", metric_path),
            ("wandb.retry_jsonl_path", retry_path),
        ):
            if path.is_absolute() or ".." in path.parts or not path.name:
                raise ValueError(f"{label} must be a repository-relative artifact file")
            try:
                path.relative_to(artifact_root)
            except ValueError:
                raise ValueError(f"{label} must be within paths.artifact_dir") from None
        if metric_path == retry_path:
            raise ValueError("local metric and W&B retry paths must differ")
        return self


def secret_environment_names(config: RuntimeConfig) -> tuple[str, ...]:
    """Return credential identifiers needed by enabled runtime features."""

    names = [config.security.modelscope_token_env]
    if config.wandb.enabled:
        names.append(config.security.wandb_api_key_env)
    return tuple(names)


def looks_like_unresolved_sentinel(value: object) -> bool:
    """Detect decision/benchmark placeholders before they reach Pydantic errors."""

    return isinstance(value, str) and bool(
        re.fullmatch(r"(?:DECISION|BENCHMARK|REQUIRED)_[A-Z0-9_]+", value)
    )
