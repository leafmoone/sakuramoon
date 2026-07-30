"""Pydantic schema for all runtime configuration surfaces."""

from __future__ import annotations

import re
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
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
Probability = Annotated[ExactFloat, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[ExactFloat, Field(gt=0.0)]
NonNegativeFloat = Annotated[ExactFloat, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
SecretEnvName = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$", min_length=2, max_length=128),
]
FixedZero = Annotated[ExactFloat, Field(ge=0.0, le=0.0)]
FixedOne = Annotated[ExactFloat, Field(ge=1.0, le=1.0)]
FixedPointOne = Annotated[ExactFloat, Field(ge=0.1, le=0.1)]
FixedPointEight = Annotated[ExactFloat, Field(ge=0.8, le=0.8)]
FixedFour = Annotated[ExactFloat, Field(ge=4.0, le=4.0)]
FixedSixteen = Annotated[ExactFloat, Field(ge=16.0, le=16.0)]
FixedThousand = Annotated[ExactFloat, Field(ge=1000.0, le=1000.0)]
FixedNormEps = Annotated[ExactFloat, Field(ge=0.000001, le=0.000001)]
FixedNegativePointEight = Annotated[ExactFloat, Field(ge=-0.8, le=-0.8)]
FixedPointZeroFive = Annotated[ExactFloat, Field(ge=0.05, le=0.05)]
FixedTwoPointNine = Annotated[ExactFloat, Field(ge=2.9, le=2.9)]
FixedLearningRate = Annotated[ExactFloat, Field(ge=0.00002, le=0.00002)]
FixedOptimizerEps = Annotated[ExactFloat, Field(ge=0.00000001, le=0.00000001)]
FixedPointZeroOne = Annotated[ExactFloat, Field(ge=0.01, le=0.01)]
FixedDecayLearningRate = Annotated[ExactFloat, Field(ge=0.000002, le=0.000002)]
FixedSix = Annotated[ExactFloat, Field(ge=6.0, le=6.0)]
FixedPointZeroTwo = Annotated[ExactFloat, Field(ge=0.02, le=0.02)]


class StrictModel(BaseModel):
    """Base for immutable, exact-type, unknown-key rejecting config tables."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class RunConfig(StrictModel):
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    stage: Literal["S0", "S1", "G1", "S2", "G2", "S3", "H1", "H2"]
    seed: NonNegativeInt


class PathsConfig(StrictModel):
    run_dir: Annotated[str, StringConstraints(min_length=1)]
    cache_dir: Annotated[str, StringConstraints(min_length=1)]
    checkpoint_dir: Annotated[str, StringConstraints(min_length=1)]
    artifact_dir: Annotated[str, StringConstraints(min_length=1)]


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
    repo_id: Literal["leafmoone/webdataset_danbooru"]
    revision: Annotated[str, StringConstraints(min_length=1)]


class DataManifestConfig(StrictModel):
    path: Annotated[str, StringConstraints(min_length=1)]
    sha256: Sha256


class DataCacheConfig(StrictModel):
    low_watermark_gib: Annotated[int, Field(ge=300, lt=500)]
    high_watermark_gib: Annotated[int, Field(gt=300, le=500)]
    download_concurrency: PositiveInt
    range_workers: PositiveInt
    persistent_workers_per_rank: PositiveInt
    ready_batches_per_rank: PositiveInt

    @model_validator(mode="after")
    def validate_watermarks(self) -> DataCacheConfig:
        if self.low_watermark_gib >= self.high_watermark_gib:
            raise ValueError("cache low watermark must be below high watermark")
        return self


class DataValidationConfig(StrictModel):
    manifest_path: Annotated[str, StringConstraints(min_length=1)]
    manifest_sha256: Sha256
    sample_count: Literal[2000]
    exclude_before_shuffle: Literal[True]


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
    validation: DataValidationConfig
    image: DataImageConfig
    buckets: DataBucketsConfig


class NlDropoutConfig(StrictModel):
    long_names: Probability
    long_no_names: Probability
    short_vibes: Probability
    nl2: Probability
    nl3: Probability

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
    general: Probability
    artist: Probability
    character: Probability
    copyright: Probability
    nsfw: Probability
    candidate_source: Probability
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
            "nsfw",
            "character",
            "copyright",
            "general",
            "nl",
        ):
            raise ValueError("caption category order differs from the approved protocol")
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
    bidirectional_attention_layers: Literal[1]
    no_positional_encoding: Literal[True]

    @model_validator(mode="after")
    def validate_blocks(self) -> TextModelConfig:
        if self.hidden_state_blocks != (2, 4, 8, 12, 16, 20, 24):
            raise ValueError("text hidden-state blocks differ from the approved mapping")
        return self


class StyleModelConfig(StrictModel):
    query_count: Literal[4]
    input_size: Literal[2048]
    hidden_size: Literal[1024]
    mlp_intermediate_size: Literal[2048]
    output_size: Literal[2560]
    null_tokens_learned: Literal[True]


class PackingModelConfig(StrictModel):
    order: StringTuple
    style_tokens: Literal[4]
    remove_text_padding: Literal[True]
    cross_sample_attention: Literal[False]

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
    interpolation: Literal["z_t=t*x+(1-t)*epsilon"]
    velocity_loss_dtype: Literal["float32"]
    reduction: Literal["per_sample_then_global_sample_mean"]


class TimestepConfig(StrictModel):
    distribution: Literal["jlt"]
    p_mean: FixedNegativePointEight
    p_std: FixedPointEight
    noise_scale: FixedOne
    t_eps: FixedPointZeroFive


class SamplingConfig(StrictModel):
    solver: Literal["heun_linear_time_final_euler"]
    steps: Literal[50]
    nfe: Literal[99]
    state_dtype: Literal["float32"]


class CfgConfig(StrictModel):
    scale: FixedTwoPointNine
    full_interval: Literal[True]
    rescale: Literal[False]
    conversion_order: Literal["each_x_to_v_then_cfg"]


class OptimizerConfig(StrictModel):
    name: Literal["torchao_adamw8bit"]
    lr: FixedLearningRate
    betas: Annotated[tuple[ExactFloat, ExactFloat], BeforeValidator(_toml_array_to_tuple)]
    eps: FixedOptimizerEps
    block_size: Literal[256]
    bf16_stochastic_round: Literal[True]
    matrix_weight_decay: FixedPointZeroOne
    sensitive_weight_decay: FixedZero

    @model_validator(mode="after")
    def validate_betas(self) -> OptimizerConfig:
        if self.betas != (0.9, 0.95):
            raise ValueError("optimizer betas differ from the approved values")
        return self


class SchedulerConfig(StrictModel):
    name: Literal["wsd"]
    warmup_updates: Literal[2000]
    stable_lr: FixedLearningRate
    decay_lr: FixedDecayLearningRate
    automatic_decay: Literal[False]


class GradientConfig(StrictModel):
    clip_norm: FixedOne
    clip_dtype: Literal["float32"]
    global_sample_mean: Literal[True]


class DistributedConfig(StrictModel):
    backend: Literal["native", "ddp"]
    world_size: Annotated[int, Field(ge=1, le=4)]
    frozen_encoders_outside_wrapper: Literal[True]
    automatic_backend_fallback: Literal[False]


class CheckpointConfig(StrictModel):
    kind: Literal["raw"]
    full_every_updates: Literal[1000]
    full_every_hours: FixedSix
    slots: PositiveInt
    atomic_complete_marker: Literal[True]
    checksum_required: Literal[True]
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
    planned_updates: PositiveInt
    manual_finalize: Literal[True]
    automatic_transition: Literal[False]

    @model_validator(mode="after")
    def validate_transition(self) -> StageConfig:
        expected: dict[str, tuple[str, int, int, int]] = {
            "S0": ("", 1, 16, 256),
            "S1": ("S0", 4, 16, 256),
            "G1": ("S1", 4, 20, 256),
            "S2": ("G1", 4, 20, 512),
            "G2": ("S2", 4, 24, 512),
            "S3": ("G2", 4, 24, 512),
            "H1": ("S3", 4, 24, 768),
            "H2": ("H1", 4, 24, 1024),
        }
        actual = (self.predecessor, self.world_size, self.depth, self.resolution)
        if actual != expected[self.name]:
            raise ValueError("stage topology does not match the approved transition graph")
        if self.name in {"H1", "H2"} and self.enabled:
            raise ValueError("H1/H2 must remain disabled until separately approved")
        return self


class KernelsConfig(StrictModel):
    attention_backend: Literal["fa4_varlen", "dense_sdpa_reference"]
    production_attention_backend: Literal["fa4_varlen"]
    reference_attention_backend: Literal["dense_sdpa_reference"]
    dtype: Literal["bfloat16"]
    native_gqa: Literal[True]
    repeat_kv_heads: Literal[False]
    silent_fallback: Literal[False]


class CompileConfig(StrictModel):
    regional_enabled: bool
    minimum_end_to_end_gain_percent: Annotated[ExactFloat, Field(ge=3.0)]
    automatic_enable: Literal[False]


class ProfilingConfig(StrictModel):
    enabled: bool
    schedule_updates: PositiveInt
    record_shapes: bool
    profile_memory: bool


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


class WandbConfig(StrictModel):
    enabled: bool
    project: Annotated[str, StringConstraints(min_length=1)]
    entity: Annotated[str, StringConstraints(min_length=1)]
    offline_on_network_error: Literal[True]


class TimingConfig(StrictModel):
    enabled: bool
    cuda_events: Literal[True]
    force_synchronize_each_phase: Literal[False]
    phases: StringTuple

    @model_validator(mode="after")
    def validate_phases(self) -> TimingConfig:
        required = {
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
        }
        if set(self.phases) != required or len(self.phases) != len(required):
            raise ValueError("timing phases must contain each required phase exactly once")
        return self


class FidConfig(StrictModel):
    enabled: bool
    every_successful_updates: PositiveInt
    trend_samples: PositiveInt
    acceptance_samples: PositiveInt
    feature_extractor: Annotated[str, StringConstraints(min_length=1)]
    real_stats_sha256: Sha256


class IsConfig(StrictModel):
    enabled: bool
    every_successful_updates: PositiveInt
    trend_samples: PositiveInt
    acceptance_samples: PositiveInt
    splits: PositiveInt


class EvaluationConfig(StrictModel):
    stage_end: Literal[True]
    explicit_job: Literal[True]
    sampling: SamplingConfig
    fid: FidConfig
    is_: IsConfig = Field(alias="is")


class RuntimeConfig(StrictModel):
    schema_version: Literal[1]
    run: RunConfig
    paths: PathsConfig
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
    compile: CompileConfig
    profiling: ProfilingConfig
    failure: FailureConfig
    logging: LoggingConfig
    wandb: WandbConfig
    timing: TimingConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def validate_cross_table_contract(self) -> RuntimeConfig:
        if self.run.stage != self.stage.name:
            raise ValueError("run.stage and stage.name must match")
        if not self.stage.enabled:
            raise ValueError("selected stage must be enabled")
        expected_backend = "native" if self.stage.name == "S0" else "ddp"
        if self.distributed.backend != expected_backend:
            raise ValueError("distributed backend does not match the selected stage")
        if self.distributed.world_size != self.stage.world_size:
            raise ValueError("distributed and stage world_size must match")
        if self.growth.enabled != (self.stage.name in {"G1", "G2"}):
            raise ValueError("growth is enabled only for G1 and G2")
        return self


def secret_environment_names(config: RuntimeConfig) -> tuple[str, str]:
    """Return the only credential identifiers retained by the config object."""

    return (
        config.security.modelscope_token_env,
        config.security.wandb_api_key_env,
    )


def looks_like_unresolved_sentinel(value: object) -> bool:
    """Detect decision/benchmark placeholders before they reach Pydantic errors."""

    return isinstance(value, str) and bool(
        re.fullmatch(r"(?:DECISION|BENCHMARK|REQUIRED)_[A-Z0-9_]+", value)
    )
