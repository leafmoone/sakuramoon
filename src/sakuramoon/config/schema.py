"""Config-driven runtime configuration schema.

The resolved configuration decides how training runs.  Values are validated
for real numeric/structural validity; they are not pinned to historical
stage recipes.  Legacy stage vocabulary is translated once at the load
boundary (see :mod:`sakuramoon.config.load`), never inside the runtime.
"""

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

from sakuramoon.model.slots import active_slot_ids
from sakuramoon.sampling.profiles import (
    SamplingProfile,
    SamplingSolver,
    TimeSchedule,
)


def _toml_array_to_tuple(value: object) -> object:
    if type(value) is list:
        return tuple(cast(list[object], value))
    return value


def _toml_number_to_float(value: object) -> object:
    """One-shot input-boundary coercion: TOML int/float -> Python float.

    ``1`` and ``1.0`` are equivalent legal spellings for a float field;
    booleans and strings are never coerced.
    """

    if isinstance(value, bool):
        raise ValueError("boolean is not a valid numeric value")
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("numeric value must be finite")
        return value
    if type(value) is int:
        return float(value)
    raise ValueError("value must use TOML float or integer syntax")


StringTuple = Annotated[tuple[str, ...], BeforeValidator(_toml_array_to_tuple)]
IntTuple = Annotated[tuple[int, ...], BeforeValidator(_toml_array_to_tuple)]
Float = Annotated[float, BeforeValidator(_toml_number_to_float)]
PositiveFloat = Annotated[Float, Field(gt=0.0)]
NonNegativeFloat = Annotated[Float, Field(ge=0.0)]
UnitFloat = Annotated[Float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
SecretEnvName = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$", min_length=2, max_length=128),
]


class StrictModel(BaseModel):
    """Base for immutable, exact-type, unknown-key rejecting config tables."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class RunConfig(StrictModel):
    """Run identity.  ``label`` is display-only: it never selects backend,
    depth, resolution, device count, or growth behavior."""

    run_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    intent: Literal["train", "eval", "sample", "template"]
    seed: NonNegativeInt
    label: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = (
        None
    )


class PathsConfig(StrictModel):
    """Output paths.  Relative values are interpreted against the deployment
    root (``--root``); absolute values are used as given."""

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
    """Qwen text-encoder asset.  ``layers``/``hidden_size`` describe the
    asset and are relation-checked against the text-conditioning config and
    the asset's own metadata at load time."""

    local_path: Annotated[str, StringConstraints(min_length=1)] = (
        "model/qwen_3.5_2B"
    )
    dtype: Literal["bfloat16"]
    frozen: bool = True
    layers: PositiveInt = 24
    hidden_size: PositiveInt = 2048
    use_cache: bool = False
    visual_path_enabled: bool = False


class VaeAssetConfig(StrictModel):
    local_path: Annotated[str, StringConstraints(min_length=1)] = "model/vae"
    dtype: Literal["bfloat16"]
    frozen: bool = True
    latent_channels: PositiveInt = 128
    downsample_factor: PositiveInt = 16
    sample_posterior: bool = False


class AssetsConfig(StrictModel):
    qwen: QwenAssetConfig
    vae: VaeAssetConfig


class DataSourceConfig(StrictModel):
    repo_id: Annotated[str, StringConstraints(min_length=1)]
    revision: Annotated[str, StringConstraints(min_length=1)]


class DataManifestConfig(StrictModel):
    path: Annotated[str, StringConstraints(min_length=1)]
    initialize_if_missing: bool = True
    refresh_existing: bool = False

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
    low_watermark_gib: NonNegativeInt
    high_watermark_gib: PositiveInt
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
    socket_path: Annotated[str, StringConstraints(min_length=1)] = (
        "/run/sakuramoon/data-service.sock"
    )
    ownership_lock_path: Annotated[str, StringConstraints(min_length=1)] = (
        "/run/sakuramoon/data-service.lock"
    )
    mainset_path: Annotated[str, StringConstraints(min_length=1)]
    request_timeout_seconds: PositiveFloat
    lease_channel_capacity: PositiveInt
    ack_channel_capacity: PositiveInt


class DataTransportConfig(StrictModel):
    connect_timeout_seconds: PositiveFloat
    read_timeout_seconds: PositiveFloat
    max_retries: NonNegativeInt
    retry_backoff_seconds: NonNegativeFloat
    # A streamed chunk is carried inside one service protocol frame, so the
    # protocol frame bound (data.service_protocol.MAX_SERVICE_FRAME_BYTES)
    # stays a hard capability limit.
    stream_chunk_bytes: Annotated[int, Field(ge=65536, le=16777216)]
    streams_per_shard: PositiveInt


class DataLoaderConfig(StrictModel):
    pin_memory: bool
    drop_last: bool
    length_sort_window_batches: PositiveInt = 1


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
    exif_transpose: bool = True
    color_mode: Literal["RGB"]
    no_upscale: bool = True
    preserve_aspect_ratio: bool = True
    allow_padding: bool = False
    min_crop_retention: Annotated[Float, Field(gt=0.0, le=1.0)]


class DataBucketsConfig(StrictModel):
    """Aspect-bucket geometry.  The bucket list is derived from these values
    at data-assembly time (see data/buckets.py); no fixed count applies."""

    base_area_px: PositiveInt
    quantum_px: PositiveInt
    min_short_edge_px: PositiveInt
    max_aspect_ratio: PositiveFloat
    transpose_closed: bool = True

    @model_validator(mode="after")
    def validate_geometry(self) -> DataBucketsConfig:
        if self.base_area_px < self.quantum_px * self.quantum_px:
            raise ValueError("buckets base_area_px must cover at least one quantum tile")
        if self.min_short_edge_px % self.quantum_px:
            raise ValueError(
                "buckets min_short_edge_px must be a multiple of quantum_px"
            )
        if self.max_aspect_ratio < 1.0:
            raise ValueError("buckets max_aspect_ratio must be >= 1.0")
        return self


class DataSpatialCropConfig(StrictModel):
    """Strict shifted-bucket spatial crop policy (zoom/shift training data)."""

    enabled: bool
    mode: Literal["shifted_bucket"]
    probability: UnitFloat
    min_equivalent_zoom: PositiveFloat
    max_equivalent_zoom: PositiveFloat
    zoom_distribution: Literal["sqrt_uniform_high"]
    offset_distribution: Literal["uniform_independent"]
    fallback_to_aspect_bucket: bool = True

    @model_validator(mode="after")
    def validate_spatial_crop(self) -> DataSpatialCropConfig:
        if self.min_equivalent_zoom >= self.max_equivalent_zoom:
            raise ValueError(
                "spatial crop min_equivalent_zoom must be below max_equivalent_zoom"
            )
        if self.min_equivalent_zoom <= 1.0:
            raise ValueError("spatial crop min_equivalent_zoom must be above 1.0")
        if self.enabled and self.probability <= 0.0:
            raise ValueError("spatial crop probability must be positive when enabled")
        if not self.enabled and self.probability != 0.0:
            raise ValueError("spatial crop probability must be zero when disabled")
        return self


class DataTransparentBackgroundConfig(StrictModel):
    """Transparent-background white-composite policy (data-strategy only)."""

    enabled: bool
    trigger_tag: Annotated[str, StringConstraints(min_length=1)] = (
        "transparent_background"
    )
    replacement_tag: Annotated[str, StringConstraints(min_length=1)] = (
        "white_background"
    )
    composite_color: Literal["white"] = "white"
    conflict_background_tags: StringTuple = ()
    special_alpha_tags: StringTuple = ()

    @model_validator(mode="after")
    def validate_transparent_background(self) -> DataTransparentBackgroundConfig:
        if self.trigger_tag == self.replacement_tag:
            raise ValueError(
                "transparent_background trigger_tag and replacement_tag must differ"
            )
        for tag in (*self.conflict_background_tags, *self.special_alpha_tags):
            if (
                type(tag) is not str
                or not tag
                or tag != tag.strip()
                or "\n" in tag
                or "\r" in tag
                or "\0" in tag
            ):
                raise ValueError(
                    "transparent_background tag sets must be non-empty trim-stable strings"
                )
        return self


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
    spatial_crop: DataSpatialCropConfig
    transparent_background: DataTransparentBackgroundConfig

    @model_validator(mode="after")
    def validate_spatial_crop_retention(self) -> DataConfig:
        max_zoom = self.spatial_crop.max_equivalent_zoom
        if 1.0 / max_zoom**2 < self.image.min_crop_retention:
            raise ValueError(
                "spatial crop max_equivalent_zoom violates the "
                "min_crop_retention guard (1/max_zoom**2 must reach it exactly)"
            )
        return self

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
    """Per-family NL dropout probabilities.  Each family is independent;
    all five are applied by the caption routing (data/caption.py)."""

    long_names: UnitFloat
    long_no_names: UnitFloat
    short_vibes: UnitFloat
    nl2: UnitFloat
    nl3: UnitFloat


class CaptionDropoutConfig(StrictModel):
    all_condition: UnitFloat
    condition_route: UnitFloat
    condition_only: UnitFloat
    tag: UnitFloat
    candidate_source: UnitFloat
    nl: NlDropoutConfig

    @model_validator(mode="after")
    def validate_route_mass(self) -> CaptionDropoutConfig:
        if self.condition_route + self.condition_only > 1.0:
            raise ValueError(
                "condition route and condition-only dropout probabilities "
                "must not exceed 1.0 together"
            )
        return self


class CaptionConfig(StrictModel):
    category_order: StringTuple
    condition_mode: Literal["artist", "artist_or_character"]
    tag_separator: Literal[", "]
    tag_nl_separator: Literal["\n\n"]
    text_condition_max: PositiveInt
    condition_buckets: IntTuple
    qwen_dense_lengths: IntTuple
    dropout: CaptionDropoutConfig

    @model_validator(mode="after")
    def validate_protocol_sequences(self) -> CaptionConfig:
        if self.category_order != ("tags", "nl", "condition"):
            raise ValueError(
                "caption category order differs from the approved protocol"
            )
        for name, values in (
            ("condition_buckets", self.condition_buckets),
            ("qwen_dense_lengths", self.qwen_dense_lengths),
        ):
            if (
                not values
                or any(type(v) is not int or v <= 0 for v in values)
                or any(later <= earlier for earlier, later in zip(values, values[1:]))
            ):
                raise ValueError(
                    f"caption {name} must be strictly increasing positive integers"
                )
        if self.condition_buckets[-1] > self.text_condition_max:
            raise ValueError(
                "caption condition_buckets must stay within text_condition_max"
            )
        return self


class TextModelConfig(StrictModel):
    """Fused text-conditioning transformer over Qwen hidden states.

    ``hidden_state_blocks`` selects the layer subset; its values are 1-based
    Qwen block indices and are relation-checked against the Qwen asset
    (RuntimeConfig) and the Qwen asset metadata at load time."""

    hidden_state_blocks: IntTuple
    input_size: PositiveInt
    adapter_size: PositiveInt
    output_size: PositiveInt
    groups: PositiveInt
    attention_heads: PositiveInt
    bidirectional_attention_layers: PositiveInt
    no_positional_encoding: bool
    norm_eps: PositiveFloat
    mix_gate_init: NonNegativeFloat
    layer_scale_init: PositiveFloat
    projection_bias: bool
    linear_dtype: Literal["bfloat16"]
    sensitive_dtype: Literal["float32"]

    @model_validator(mode="after")
    def validate_blocks(self) -> TextModelConfig:
        blocks = self.hidden_state_blocks
        if (
            not blocks
            or any(type(b) is not int or b < 1 for b in blocks)
            or any(later <= earlier for earlier, later in zip(blocks, blocks[1:]))
        ):
            raise ValueError(
                "text hidden_state_blocks must be strictly increasing positive block indices"
            )
        return self


class ConditionTokensModelConfig(StrictModel):
    token_count: PositiveInt
    input_size: PositiveInt
    hidden_size: PositiveInt
    mlp_intermediate_size: PositiveInt
    output_size: PositiveInt
    null_tokens_learned: bool
    attention_heads: PositiveInt
    norm_eps: PositiveFloat
    init_std: PositiveFloat
    projection_bias: bool
    linear_dtype: Literal["bfloat16"]
    sensitive_dtype: Literal["float32"]


class PackingModelConfig(StrictModel):
    order: StringTuple
    remove_text_padding: bool
    modality_init_std: PositiveFloat

    @model_validator(mode="after")
    def validate_order(self) -> PackingModelConfig:
        if self.order != ("text", "condition", "image"):
            raise ValueError("packing order must be the implemented protocol order")
        return self


class RopeModelConfig(StrictModel):
    head_dim: PositiveInt
    nope_dim: PositiveInt
    y_dim: PositiveInt
    x_dim: PositiveInt
    position_scale: PositiveFloat
    theta: PositiveFloat
    cell_center: bool
    area_normalized: bool

    @model_validator(mode="after")
    def validate_split(self) -> RopeModelConfig:
        if (
            self.nope_dim + self.y_dim + self.x_dim != self.head_dim
            or self.y_dim != self.x_dim
            or self.y_dim % 2
        ):
            raise ValueError(
                "rope dimensions must satisfy nope+y+x == head_dim with an even y == x"
            )
        return self


class DitModelConfig(StrictModel):
    """DiT core.  ``depth`` and ``active_slot_ids`` describe the real block
    topology: either may be given (the other is derived); when both are
    given they must agree in count, ids must be unique and non-negative, and
    any id range is legal.  Slot names keep the checkpoint's original
    numbering on resume (see checkpoint/schema.py)."""

    depth: PositiveInt | None = None
    active_slot_ids: IntTuple | None = None
    hidden_size: PositiveInt
    head_dim: PositiveInt
    q_heads: PositiveInt
    kv_heads: PositiveInt
    intermediate_size: PositiveInt
    attention_dropout: UnitFloat
    mlp_dropout: UnitFloat
    projection_bias: bool
    norm_eps: PositiveFloat
    norm_accumulation: Literal["float32"]
    activation_dtype: Literal["bfloat16"]

    @model_validator(mode="before")
    @classmethod
    def derive_topology(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        payload = cast(dict[str, object], dict(value))
        depth = payload.get("depth")
        slots = payload.get("active_slot_ids")
        if type(slots) is list:
            slots = tuple(cast(list[object], slots))
        if depth is None and slots is None:
            raise ValueError("model.dit requires depth or active_slot_ids")
        if depth is not None and slots is None:
            # Historical depths keep their established slot topology (see
            # model.slots); other depths use the contiguous id range.
            resolved_slots: object = list(active_slot_ids(cast(int, depth)))
        else:
            assert slots is not None
            if depth is None:
                payload["depth"] = len(cast(tuple[object, ...], slots))
            elif len(cast(tuple[object, ...], slots)) != cast(int, depth):
                raise ValueError(
                    "model.dit active_slot_ids count must equal depth"
                )
            resolved_slots = slots
        payload["active_slot_ids"] = resolved_slots
        return payload

    @model_validator(mode="after")
    def validate_topology(self) -> DitModelConfig:
        slots = cast(tuple[int, ...], self.active_slot_ids)
        if (
            not slots
            or any(type(s) is not int or s < 0 for s in slots)
            or len(set(slots)) != len(slots)
        ):
            raise ValueError(
                "model.dit active_slot_ids must be unique non-negative ids"
            )
        if self.q_heads % self.kv_heads:
            raise ValueError("model.dit q_heads must be a multiple of kv_heads")
        return self


class ConditionModelConfig(StrictModel):
    timestep_dim: PositiveInt
    size_dim: PositiveInt
    aspect_dim: PositiveInt
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    block_modulation_chunks: PositiveInt
    shared_projection_zero_init: bool
    per_block_bias_zero_init: bool

    @model_validator(mode="after")
    def validate_input_dim(self) -> ConditionModelConfig:
        if self.input_dim != self.timestep_dim + self.size_dim + self.aspect_dim:
            raise ValueError(
                "condition input_dim must equal timestep+size+aspect dims"
            )
        return self


class HeadModelConfig(StrictModel):
    final_modulation_size: PositiveInt
    out_channels: PositiveInt
    prediction_type: Literal["x"]
    image_span_only: bool
    weight_zero_init: bool
    bias_zero_init: bool


class ModelConfig(StrictModel):
    text: TextModelConfig
    condition_tokens: ConditionTokensModelConfig
    packing: PackingModelConfig
    rope: RopeModelConfig
    dit: DitModelConfig
    condition: ConditionModelConfig
    head: HeadModelConfig

    @model_validator(mode="after")
    def validate_relations(self) -> ModelConfig:
        if self.text.output_size != self.dit.hidden_size:
            raise ValueError("text output_size must match the DiT hidden_size")
        if self.condition_tokens.output_size != self.dit.hidden_size:
            raise ValueError(
                "condition_tokens output_size must match the DiT hidden_size"
            )
        if self.rope.head_dim != self.dit.head_dim:
            raise ValueError("rope head_dim must match the DiT head_dim")
        if self.head.final_modulation_size != 2 * self.dit.hidden_size:
            raise ValueError(
                "head final_modulation_size must be twice the DiT hidden_size"
            )
        return self


class ObjectiveConfig(StrictModel):
    """Implemented objective identity.  These literals name the single
    implemented formulas; they are capability boundaries, not tuning knobs."""

    prediction_type: Literal["x"]
    loss: Literal["jlt_x_prediction_velocity_mse"]
    target_velocity: Literal["x_to_v(clean,state,t,t_eps)"]
    endpoint_weighting: Literal["inverse_square_clamped"]
    interpolation: Literal["z_t=t*x+(1-t)*epsilon"]
    velocity_loss_dtype: Literal["float32"]
    reduction: Literal["per_sample_then_global_sample_mean"]


class TimestepConfig(StrictModel):
    """JLT timestep sampling and noise scaling.  Values feed the formulas
    directly (objective/flow.py); the sigmoid-normal form is preserved."""

    distribution: Literal["jlt"]
    p_mean: Float
    p_std: PositiveFloat
    noise_scale: PositiveFloat
    t_eps: Annotated[Float, Field(gt=0.0, lt=1.0)]


class SamplingProfileConfig(StrictModel):
    solver: SamplingSolver
    steps: PositiveInt
    time_schedule: TimeSchedule


class TrainingSamplingConfig(StrictModel):
    """Periodic image samples made from captions seen by the training loop.

    ``fixed_cohort=locked`` keeps the historical locked suite as an optional
    preset; its image count is derived from the suite content, not forced."""

    enabled: bool = True
    every_updates: PositiveInt = 1000
    image_count: PositiveInt = 12
    output_subdir: Annotated[str, StringConstraints(min_length=1)] = "sample"
    fixed_cohort: Literal["neutral", "locked"] = "neutral"
    longitudinal_pin_update: PositiveInt | None = None

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
    """Sampling is profile-driven.  ``profiles`` holds any number of named
    presets (the historical preview/balanced/reference remain the default
    template presets); ``profile`` selects one by name.  NFE is computed
    from the selected solver/steps."""

    profile: Annotated[str, StringConstraints(min_length=1)]
    profiles: dict[str, SamplingProfileConfig]
    state_dtype: Literal["float32"]
    training: TrainingSamplingConfig = TrainingSamplingConfig()

    @model_validator(mode="after")
    def validate_selection(self) -> SamplingConfig:
        if self.profile not in self.profiles:
            raise ValueError(
                f"sampling profile {self.profile!r} is not defined in [sampling.profiles]"
            )
        return self

    def by_name(self, name: str) -> SamplingProfile:
        """Resolve a named profile (e.g. the evaluation selection)."""

        entry = self.profiles.get(name)
        if entry is None:
            raise ValueError(
                f"sampling profile {name!r} is not defined in [sampling.profiles]"
            )
        return SamplingProfile(
            name=name,
            solver=entry.solver,
            steps=entry.steps,
            time_schedule=entry.time_schedule,
        )

    @property
    def selected(self) -> SamplingProfile:
        return self.by_name(self.profile)

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
    """Classifier-free guidance.  ``scale`` feeds every sampling/eval path;
    the x-to-v conversion and CFG ordering stay as implemented."""

    scale: PositiveFloat
    conversion_order: Literal["each_x_to_v_then_cfg"]


class CMuonNSConfig(StrictModel):
    """Per-role Newton-Schulz depth for the hybrid CMuon optimizer.

    ``default`` applies to every semantic role; the optional per-role fields
    override it. ``canonical_map()`` resolves the full role -> depth map.
    Only used when optimizer.name == "hybrid_cmuon".
    """

    default: PositiveInt = 5
    attention_q: PositiveInt | None = None
    attention_k: PositiveInt | None = None
    attention_v: PositiveInt | None = None
    attention_content_gate: PositiveInt | None = None
    attention_out: PositiveInt | None = None
    ffn_in: PositiveInt | None = None
    ffn_down: PositiveInt | None = None
    adaln_shared: PositiveInt | None = None

    def canonical_map(self) -> dict[str, int]:
        data = self.model_dump()
        base = cast(int, data.pop("default"))
        return {
            role: (cast(int, value) if value is not None else base)
            for role, value in data.items()
        }


# Mirrors sakuramoon.optim.cmuon.CMUON_ROLES (the canonical semantic-role
# vocabulary for the hybrid CMuon optimizer).
_CMUON_TELEMETRY_ROLES = (
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_content_gate",
    "attention_out",
    "ffn_in",
    "ffn_down",
    "adaln_shared",
)


class CMuonNSTelemetryConfig(StrictModel):
    """Opt-in NS safety telemetry for the hybrid CMuon optimizer."""

    enabled: bool = False
    log_every_n: PositiveInt = 100
    roles: StringTuple = ()

    @model_validator(mode="after")
    def validate_roles(self) -> CMuonNSTelemetryConfig:
        unknown = set(self.roles) - set(_CMUON_TELEMETRY_ROLES)
        if unknown:
            raise ValueError(f"unknown cmuon telemetry roles: {sorted(unknown)}")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("cmuon telemetry roles must be unique")
        return self


class CMuonForensicConfig(StrictModel):
    """Fail-closed forensic instrumentation for the hybrid CMuon optimizer."""

    enabled: bool = False
    ring_size: PositiveInt = 10
    ceiling_multiplier: PositiveFloat = 10.0
    max_abs_learn_steps: PositiveInt = 50
    max_abs_alarm_mult: PositiveFloat = 20.0
    dump_dir: str | None = None
    divergence_rel_tol: PositiveFloat = 1e-1


class CMuonGuardConfig(StrictModel):
    """Pre-NS low-signal guard for the guarded canonical CMuon candidate.

    Used only when optimizer.name == "hybrid_cmuon_guarded_canonical" (or the
    FP32-rescue candidate, which inherits the reference-table contract).
    All values are calibration-derived (there are NO preset defaults).
    """

    enabled: bool = True
    guard_ratio: PositiveFloat
    reference_decay: Annotated[Float, Field(gt=0.0, le=1.0)]
    min_reference: PositiveFloat
    numerical_floor: PositiveFloat
    warmup_observations: NonNegativeInt
    invariant_check: bool = True
    references: dict[str, float] = {}


class OptimizerConfig(StrictModel):
    name: Literal[
        "torchao_adamw8bit",
        "hybrid_cmuon",
        "hybrid_cmuon_guarded_canonical",
        "hybrid_cmuon_canonical_ns4_fp32_rescue",
    ]
    base_lr: PositiveFloat
    reference_batch: PositiveInt
    lr_scaling: Literal["linear_global_batch", "none"]
    betas: Annotated[tuple[Float, Float], BeforeValidator(_toml_array_to_tuple)]
    eps: PositiveFloat
    # Quantization block size is part of the stored optimizer-state format;
    # the checkpoint reader rejects a mismatch against the loaded state.
    block_size: PositiveInt
    bf16_stochastic_round: bool
    matrix_weight_decay: UnitFloat
    sensitive_weight_decay: UnitFloat
    cmuon_ns_steps: PositiveInt = 5
    cmuon_ns: CMuonNSConfig | None = None
    cmuon_momentum_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    cmuon_chunk_rescale_sqrt_n: bool = False
    cmuon_ns_telemetry: CMuonNSTelemetryConfig | None = None
    cmuon_forensic: CMuonForensicConfig | None = None
    cmuon_routing_exclude: StringTuple = ()
    cmuon_guard: CMuonGuardConfig | None = None

    @model_validator(mode="after")
    def validate_betas(self) -> OptimizerConfig:
        for beta in self.betas:
            if not 0.0 <= beta < 1.0:
                raise ValueError("optimizer betas must lie in [0,1)")
        return self

    @model_validator(mode="after")
    def validate_cmuon_guard(self) -> OptimizerConfig:
        if self.name in (
            "hybrid_cmuon_guarded_canonical",
            "hybrid_cmuon_canonical_ns4_fp32_rescue",
        ):
            if self.cmuon_guard is None or not self.cmuon_guard.enabled:
                raise ValueError(
                    f"{self.name} requires an enabled [optimizer.cmuon_guard] "
                    "section"
                )
            if not self.cmuon_guard.references:
                raise ValueError(
                    "[optimizer.cmuon_guard] references must hold the "
                    "calibration bootstrap table (P3 artifact)"
                )
        elif self.cmuon_guard is not None:
            raise ValueError(
                "[optimizer.cmuon_guard] is only valid for optimizer.name = "
                '"hybrid_cmuon_guarded_canonical" or '
                '"hybrid_cmuon_canonical_ns4_fp32_rescue"'
            )
        return self

    @model_validator(mode="after")
    def validate_cmuon_routing_exclude(self) -> OptimizerConfig:
        unknown = set(self.cmuon_routing_exclude) - set(_CMUON_TELEMETRY_ROLES)
        if unknown:
            raise ValueError(f"unknown cmuon routing exclude roles: {sorted(unknown)}")
        if len(set(self.cmuon_routing_exclude)) != len(self.cmuon_routing_exclude):
            raise ValueError("cmuon routing exclude roles must be unique")
        return self


class SchedulerConfig(StrictModel):
    """LR schedule over the successful-update global coordinate.

    ``linear_warmup_constant`` warms up linearly then holds the scaled LR;
    ``constant`` holds the scaled LR from update 0 (no warmup).  A resume
    never restarts the warmup implicitly: the loop continues from the stored
    update coordinate; a fresh warmup requires an explicit config change.
    """

    name: Literal["linear_warmup_constant", "constant"]
    warmup_updates: NonNegativeInt

    @model_validator(mode="after")
    def validate_warmup(self) -> SchedulerConfig:
        if self.name == "constant" and self.warmup_updates != 0:
            raise ValueError(
                "scheduler constant requires warmup_updates = 0 "
                "(use linear_warmup_constant for a warmup phase)"
            )
        return self


class GradientConfig(StrictModel):
    """Gradient clipping.  ``clip_norm = 0.0`` disables clipping explicitly;
    nonfinite checks and the sample-mean gradient normalization are
    unconditional invariants of the update step."""

    clip_norm: NonNegativeFloat
    clip_dtype: Literal["float32"]


class DistributedConfig(StrictModel):
    """Topology request.  ``world_size`` is the launcher request; after
    process-group initialization the actual group is the source of truth
    (the runtime reports a concrete error on mismatch).  Device legality is
    decided by the launcher, the process group, the kernels, and the data
    service — not by this table."""

    backend: Literal["native", "accelerate", "ddp"]
    world_size: PositiveInt


class CheckpointConfig(StrictModel):
    kind: Literal["raw"]
    full_every_updates: PositiveInt
    slots: PositiveInt
    atomic_complete_marker: bool
    canonical_fqn: bool


class GrowthConfig(StrictModel):
    """Block growth.  ``enabled`` starts (or declares) a growth run; the new
    slot set is the difference between the target ``model.dit`` topology and
    the source checkpoint's slots (an explicit resume keeps the checkpoint's
    in-flight anchor; completed growth stays alpha=1).  ``ramp_updates`` is
    the half-cosine ramp length applied when a growth starts fresh."""

    enabled: bool = False
    alpha_schedule: Literal["half_cosine"] = "half_cosine"
    ramp_updates: PositiveInt = 1000
    init_strategy: Literal["random"] = "random"


class TrainConfig(StrictModel):
    """Current-run training settings (replaces the legacy [stage] table).

    ``max_updates`` is the absolute successful-update terminal of the loop:
    the loop checks ``current_update < max_updates`` before starting each
    update, and a terminal at/below the restored update count ends the run
    with zero updates.  ``global_batch``, when present, is a consistency
    assertion only — the effective batch is always
    ``local_batch * accumulation * world_size``."""

    resolution: PositiveInt
    local_batch: PositiveInt
    accumulation: PositiveInt
    max_updates: PositiveInt
    activation_checkpoint_mode: Literal["none", "alternating", "all"]
    global_batch: PositiveInt | None = None


class KernelsConfig(StrictModel):
    attention_backend: Literal["dense_sdpa_reference", "das_fa2_varlen"]
    qwen_attention_backend: Literal["sdpa", "flash_attention_2"] = "sdpa"
    tunableop_enabled: bool = False
    tunableop_tuning: bool = False
    tunableop_record_untuned: bool = False
    tunableop_max_tuning_duration_ms: PositiveInt = 50
    torch_compile_enabled: bool = False
    torch_compile_backend: Literal["inductor"] = "inductor"
    torch_compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune-no-cudagraphs"
    ] = "default"
    torch_compile_dynamic: bool = False
    vae_torch_compile: bool = False
    dtype: Literal["bfloat16"]
    native_gqa: bool
    repeat_kv_heads: bool
    silent_fallback: bool

    @model_validator(mode="after")
    def validate_attention_invariants(self) -> KernelsConfig:
        if self.silent_fallback:
            raise ValueError(
                "kernels silent_fallback must stay disabled: unsupported "
                "attention backends fail loudly instead of falling back"
            )
        if self.repeat_kv_heads:
            raise ValueError(
                "kernels repeat_kv_heads must stay disabled: the runtime "
                "uses true GQA layouts only"
            )
        return self


class LoggingConfig(StrictModel):
    local_jsonl_path: Annotated[str, StringConstraints(min_length=1)]
    flush_every_updates: PositiveInt
    async_remote: bool = True
    noise_observation_boundary: Annotated[Float, Field(gt=0.0, lt=1.0)]
    observer_queue_capacity: PositiveInt
    observer_event_timeout_seconds: PositiveFloat


class WandbConfig(StrictModel):
    enabled: bool
    project: Annotated[str, StringConstraints(min_length=1)]
    entity: Annotated[str, StringConstraints(min_length=1)]
    offline_on_network_error: bool
    retry_jsonl_path: Annotated[str, StringConstraints(min_length=1)]
    queue_capacity: PositiveInt
    replay_retry_on_start: bool = True
    finish_on_close: bool
    resume_policy: Literal["allow"]


class TimingConfig(StrictModel):
    """Phase timing.  ``enabled = false`` runs with no-op timers: no CUDA
    events, no forced synchronization, and unmeasured phases are omitted
    from telemetry (never reported as 0 seconds).  The phase vocabulary has
    a single source of truth (telemetry/metrics.py)."""

    enabled: bool


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
    sampling_profile: Annotated[str, StringConstraints(min_length=1)]
    concept_suite_enabled: bool = False
    concept_suite_manifest: Annotated[str, StringConstraints(min_length=1)] = (
        "data/concept-benchmarks/concept-120-v1/manifest-120-v1-refs341b.json"
    )
    concept_suite_refs_root: Annotated[str, StringConstraints(min_length=1)] | None = (
        None
    )

    @model_validator(mode="before")
    @classmethod
    def default_real_sample_count(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        payload = cast(dict[str, object], value)
        if "real_sample_count" in payload or "sample_count" not in payload:
            return payload
        updated: dict[str, object] = dict(payload)
        updated["real_sample_count"] = payload["sample_count"]
        return updated

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
        for name, value in (
            ("output_dir", self.output_dir),
            ("concept_suite_manifest", self.concept_suite_manifest),
            ("concept_suite_refs_root", self.concept_suite_refs_root),
        ):
            if value is None:
                continue
            path = PurePosixPath(value)
            if not path.parts or ".." in path.parts:
                raise ValueError(
                    f"evaluation {name} must not traverse parent directories"
                )
        for name, value in (
            ("prompt_path", self.prompt_path),
            ("validation_shard_root", self.validation_shard_root),
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
                    f"evaluation {name} must be a normalized repository-relative path"
                )
        return self

    @property
    def resolved_real_sample_count(self) -> int:
        return self.real_sample_count


EvaluationConfig = Annotated[
    EvaluationDisabledConfig | EvaluationEnabledConfig,
    Field(discriminator="enabled"),
]


class IRepaConfig(StrictModel):
    """SakuraMoon iREPA v1 training-only auxiliary alignment.

    ``[irepa]`` absent = feature absent (no projector, no optimizer or model
    change).  ``enabled = false`` parses but creates no training parameters.
    ``enabled = true`` constructs the projector on the trunk; OFF->ON starts
    a fresh ramp anchored at the source update, ON->ON restores the
    projector and anchor, ON->OFF drops only the auxiliary parameters.  The
    teacher family is a real capability boundary: only the PE-Spatial
    layout is supported (checked in assets/pe_spatial.py).
    """

    enabled: bool
    teacher_id: Annotated[str, StringConstraints(min_length=1)] = (
        "facebook/PE-Spatial-B16-512"
    )
    teacher_local_path: Annotated[str, StringConstraints(min_length=1)] = (
        "model/pe_spatial_b16_512"
    )
    tap_slot: PositiveInt
    projector_kernel_size: PositiveInt
    spatial_norm: Literal["zscore"]
    spatial_norm_gamma: Annotated[Float, Field(gt=0.0, lt=1.0)] = 0.6
    spatial_norm_eps: PositiveFloat = 0.000001
    loss: Literal["cosine"]
    weight: NonNegativeFloat = 0.5
    ramp_in_updates: PositiveInt = 1000
    ramp_out_after_updates: PositiveInt | None = None
    ramp_out_updates: PositiveInt = 1000

    @model_validator(mode="after")
    def validate_teacher_path(self) -> IRepaConfig:
        path = PurePosixPath(self.teacher_local_path)
        if (
            self.teacher_local_path != self.teacher_local_path.strip()
            or "\\" in self.teacher_local_path
            or ".." in path.parts
            or path.as_posix() != self.teacher_local_path
            or not path.name
        ):
            raise ValueError(
                "irepa teacher_local_path must be a normalized "
                "repository-relative or absolute directory"
            )
        return self

    @model_validator(mode="after")
    def validate_kernel(self) -> IRepaConfig:
        if self.projector_kernel_size % 2 == 0:
            raise ValueError("irepa projector_kernel_size must be odd")
        return self

    @model_validator(mode="after")
    def validate_ramp_schedule(self) -> IRepaConfig:
        if (
            self.ramp_out_after_updates is not None
            and self.ramp_out_after_updates <= self.ramp_in_updates
        ):
            raise ValueError(
                "irepa ramp_out_after_updates must be greater than "
                "ramp_in_updates"
            )
        return self


class RuntimeConfig(StrictModel):
    schema_version: Literal[1]
    run: RunConfig
    paths: PathsConfig
    security: SecurityConfig
    assets: AssetsConfig
    data: DataConfig
    caption: CaptionConfig
    model: ModelConfig
    train: TrainConfig
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
    kernels: KernelsConfig
    logging: LoggingConfig
    wandb: WandbConfig
    timing: TimingConfig
    evaluation: EvaluationConfig
    irepa: IRepaConfig | None = None

    @model_validator(mode="after")
    def validate_train_batch(self) -> RuntimeConfig:
        if self.train.global_batch is not None and self.train.global_batch != (
            self.effective_global_batch()
        ):
            raise ValueError(
                "train.global_batch must equal local_batch * accumulation * "
                f"world_size ({self.effective_global_batch()}); "
                "remove it or fix the mismatch"
            )
        return self

    @model_validator(mode="after")
    def validate_model_relations(self) -> RuntimeConfig:
        if self.assets.qwen.hidden_size != self.model.text.input_size:
            raise ValueError(
                "assets.qwen.hidden_size must match model.text.input_size"
            )
        if max(self.model.text.hidden_state_blocks) > self.assets.qwen.layers:
            raise ValueError(
                "model.text.hidden_state_blocks must stay within "
                "assets.qwen.layers"
            )
        if self.model.head.out_channels != self.assets.vae.latent_channels:
            raise ValueError(
                "model.head.out_channels must match assets.vae.latent_channels"
            )
        return self

    @model_validator(mode="after")
    def validate_evaluation_profile(self) -> RuntimeConfig:
        evaluation = self.evaluation
        if evaluation.enabled and evaluation.sampling_profile not in self.sampling.profiles:
            raise ValueError(
                "evaluation.sampling_profile must name a [sampling.profiles] entry"
            )
        return self

    def effective_global_batch(self) -> int:
        """The single source of truth for the effective global batch."""

        return (
            self.train.local_batch
            * self.train.accumulation
            * self.distributed.world_size
        )

    def scaled_learning_rate(self) -> float:
        """Return the configured LR mode applied to the effective batch.

        ``linear_global_batch``: base_lr * global_batch / reference_batch.
        ``none``: base_lr is the absolute LR.  Resolution changes never add
        a hidden scale factor.
        """

        if self.optimizer.lr_scaling == "none":
            return self.optimizer.base_lr
        return (
            self.optimizer.base_lr
            * self.effective_global_batch()
            / self.optimizer.reference_batch
        )


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
