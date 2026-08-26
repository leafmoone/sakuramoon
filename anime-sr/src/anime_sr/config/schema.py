"""Frozen config schema for AnimeSR-Mage-UFlow (plan v2.0, 2026-08-26).

All training hyper-parameters live in ``anime-sr/config/*.toml`` (repo rule:
训练参数从 config 读取, 不在代码里复制). The schema is the *complete* v1.0
contract; loaders validate strictly (unknown fields rejected) so a config
can never silently drift from the frozen plan (docs/plan-v2.0.md §21).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base: reject unknown fields so TOML typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# project / model
# ---------------------------------------------------------------------------
class ProjectSpec(_Frozen):
    name: str = "anime-sr-mage-uflow"
    task: str = "blind-anime-sr"
    scale: int = 4
    text_condition: bool = False
    cfg: bool = False
    t2i_backbone: bool = False


class PixelEncoderSpec(_Frozen):
    """Native LQ pixel condition encoder (plan §6): 9-12M params."""

    in_channels: int = 3
    input_scales: list[int] = Field(default_factory=lambda: [256, 128, 64, 32, 16])
    stem_channels: int = 48
    stage_channels: list[int] = Field(default_factory=lambda: [48, 96, 128, 192, 256])
    stage_depths: list[int] = Field(default_factory=lambda: [2, 2, 3, 3, 4])
    param_budget_m: list[float] = Field(default_factory=lambda: [9.0, 12.0])


class UFlowStageSpec(_Frozen):
    grid: int
    dim: int
    depth: int
    q_heads: int
    kv_heads: int
    ffn: int
    attention: Literal["window-8", "global"]


def _default_uflow_stages() -> list[UFlowStageSpec]:
    """Plan §7.2 frozen table: enc 64/32 -> bot 16 -> dec 32/64."""
    return [
        UFlowStageSpec(grid=64, dim=384, depth=4, q_heads=6, kv_heads=2, ffn=1152, attention="window-8"),
        UFlowStageSpec(grid=32, dim=512, depth=6, q_heads=8, kv_heads=2, ffn=1536, attention="window-8"),
        UFlowStageSpec(grid=16, dim=768, depth=8, q_heads=12, kv_heads=4, ffn=2304, attention="global"),
        UFlowStageSpec(grid=32, dim=512, depth=6, q_heads=8, kv_heads=2, ffn=1536, attention="window-8"),
        UFlowStageSpec(grid=64, dim=384, depth=4, q_heads=6, kv_heads=2, ffn=1152, attention="window-8"),
    ]


class UFlowSpec(_Frozen):
    """Multi-scale U-Flow Transformer (plan §7), core 121-128M params."""

    stages: list[UFlowStageSpec] = Field(default_factory=_default_uflow_stages)
    head_dim: int = 64
    qk_norm: bool = True
    norm: Literal["rms"] = "rms"
    activation: Literal["swiglu"] = "swiglu"
    dropout: float = 0.0
    rope: Literal["continuous-2d"] = "continuous-2d"
    layerscale_init: float = 1e-3
    window_shift_pattern: Literal["normal-shifted"] = "normal-shifted"
    downsample: Literal["pixelunshuffle2-1x1"] = "pixelunshuffle2-1x1"
    upsample: Literal["1x1-pixelshuffle2"] = "1x1-pixelshuffle2"
    skip_fusion: Literal["concat-1x1"] = "concat-1x1"

    def validate_stage_geometry(self) -> None:
        """Structural sanity checks (called by ModelSpec)."""
        grids = [s.grid for s in self.stages]
        if len(self.stages) != 5 or grids != [64, 32, 16, 32, 64]:
            raise ValueError(f"U-Flow stages must be 5 with grids 64/32/16/32/64, got {grids}")
        for s in self.stages:
            if s.dim % s.q_heads or s.dim % s.kv_heads:
                raise ValueError(f"dim {s.dim} not divisible by q_heads {s.q_heads} / kv_heads {s.kv_heads}")
            if s.q_heads % s.kv_heads:
                raise ValueError(f"q_heads {s.q_heads} must be a multiple of kv_heads {s.kv_heads}")
            if s.ffn != 3 * s.dim:
                raise ValueError(f"ffn {s.ffn} must be 3*dim={3 * s.dim}")
        if any(s.dim % self.head_dim for s in self.stages):
            raise ValueError(f"head_dim {self.head_dim} must divide every stage dim")


class OutputHeadSpec(_Frozen):
    """Plan §7.4: RMSNorm -> 3x3 (384->384) -> SiLU -> 3x3 (384->128), zero-init out."""

    dim_in: int = 384
    dim_out: int = 128
    zero_init_out: bool = True


class ModelSpec(_Frozen):
    latent_channels: int = 128
    latent_downsample: int = 16
    param_budget_m: list[float] = Field(default_factory=lambda: [121.0, 128.0])
    grounding_enabled: bool = False  # decided by M0 VAE ceiling
    pixel_encoder: PixelEncoderSpec = Field(default_factory=PixelEncoderSpec)
    uflow: UFlowSpec = Field(default_factory=UFlowSpec)
    output_head: OutputHeadSpec = Field(default_factory=OutputHeadSpec)

    def validate_structure(self) -> None:
        self.uflow.validate_stage_geometry()
        if self.uflow.stages[0].dim != self.output_head.dim_in:
            raise ValueError("output head input dim must equal the 64x64 stage dim (384)")
        if self.output_head.dim_out != self.latent_channels:
            raise ValueError("output head output dim must equal latent_channels (128)")


# ---------------------------------------------------------------------------
# flow (plan §5, §9)
# ---------------------------------------------------------------------------
class FlowSpec(_Frozen):
    pred: Literal["v"] = "v"
    sigma_default: float = 0.0
    sigma_finite_bound: float = 1.0
    sigma_modes: dict[str, float] = Field(
        default_factory=lambda: {
            "faithful": 0.0,
            "balanced_low": 0.05,
            "balanced_high": 0.10,
            "experimental_max": 0.15,
        }
    )
    # §5.6 initial training mix; M3 A/B tests the three candidates.
    train_sigma_zero_fraction: float = 0.75
    train_sigma_noise_range: list[float] = Field(default_factory=lambda: [0.02, 0.15])
    solver_4step: Literal["heun"] = "heun"
    time_points: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    time_sampling: Literal["uniform"] = "uniform"


# ---------------------------------------------------------------------------
# data buckets (plan §10)
# ---------------------------------------------------------------------------
class BucketsSpec(_Frozen):
    lq_sizes: list[int] = Field(default_factory=lambda: [128, 192, 256])
    hr_multiplier: int = 4
    hr_multiple: int = 64
    lq_multiple: int = 16
    latent_divisible_by: int = 4
    area_aspect_constraint: bool = True


# ---------------------------------------------------------------------------
# training phases (plan §13, §14)
# ---------------------------------------------------------------------------
class CurriculumPhaseSpec(_Frozen):
    label: str
    fraction: float
    mix: dict[str, float]  # {"128": f, "192": f, "256": f}


def _default_phase1_curriculum() -> list[CurriculumPhaseSpec]:
    """Plan §13 M4: I-A 20% / I-B 30% / I-C 30% / I-D 20%."""
    return [
        CurriculumPhaseSpec(label="I-A", fraction=0.20, mix={"128": 1.0}),
        CurriculumPhaseSpec(label="I-B", fraction=0.30, mix={"128": 0.5, "192": 0.3, "256": 0.2}),
        CurriculumPhaseSpec(label="I-C", fraction=0.30, mix={"128": 0.2, "192": 0.3, "256": 0.5}),
        CurriculumPhaseSpec(label="I-D", fraction=0.20, mix={"128": 0.1, "192": 0.2, "256": 0.7}),
    ]


class Phase1Spec(_Frozen):
    """M4: flow-only, 10M target exposures (6M min / 16M max)."""

    exposure_target: int = 10_000_000
    exposure_min: int = 6_000_000
    exposure_max: int = 16_000_000
    curriculum: list[CurriculumPhaseSpec] = Field(default_factory=_default_phase1_curriculum)
    final_256_extra_exposures: int = 500_000
    loss: Literal["fm"] = "fm"


class LossWeights(_Frozen):
    """Plan §12.2 Phase-II loss terms (fixed weights; calibration of the *gradients*)."""

    flow: float = 1.0
    pixel: float = 1.0
    pixel_charbonnier_eps: float = 1e-3
    edge: float = 0.10
    flat: float = 0.05
    perceptual: float = 0.05
    perceptual_hr: int = 512


class GradCalibrationSpec(_Frozen):
    """Plan §12.3: 2000-update calibration windows for the Phase-II terms."""

    target_updates: int = 2000
    pixel: list[float] = Field(default_factory=lambda: [0.25, 0.50])
    edge: list[float] = Field(default_factory=lambda: [0.05, 0.15])
    flat: list[float] = Field(default_factory=lambda: [0.03, 0.10])
    perceptual: list[float] = Field(default_factory=lambda: [0.05, 0.15])


class Phase2Spec(_Frozen):
    """M5: one-step faithful, 2M target exposures (1M min / 4M max)."""

    exposure_target: int = 2_000_000
    exposure_min: int = 1_000_000
    exposure_max: int = 4_000_000
    size_mix: dict[str, float] = Field(default_factory=lambda: {"192": 0.2, "256": 0.8})
    batch_mix: dict[str, float] = Field(
        default_factory=lambda: {"random_t_flow": 0.5, "one_step_decode": 0.5}
    )
    loss: LossWeights = Field(default_factory=LossWeights)
    optimizer_lrs: dict[str, float] = Field(
        default_factory=lambda: {
            "uflow": 2e-5,
            "pixel_encoder": 5e-5,
            "decoder_adapters": 1e-4,
            "vae_base": 0.0,
        }
    )
    grad_calibration: GradCalibrationSpec = Field(default_factory=GradCalibrationSpec)


class OptimizerSpec(_Frozen):
    """Plan §14.1 (Phase I)."""

    name: Literal["adamw"] = "adamw"
    lr: float = 0.00015
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.05
    no_decay: list[str] = Field(
        default_factory=lambda: ["norm", "bias", "layerscale", "position", "gates"]
    )


class SchedulerSpec(_Frozen):
    warmup_fraction: float = 0.03
    type: Literal["cosine"] = "cosine"
    min_lr_ratio: float = 0.10


class GradientSpec(_Frozen):
    clip_norm: float = 1.0


class EMASpec(_Frozen):
    """Plan §14.3: EMA by *samples* (half-life 500k), not by update count."""

    half_life_samples: int = 500_000


# ---------------------------------------------------------------------------
# hardware (plan §15)
# ---------------------------------------------------------------------------
class HardwareSpec(_Frozen):
    dtype: Literal["bf16"] = "bf16"
    ddp: bool = True
    fsdp: bool = False
    attention_backend: str = "sdpa-correctness"
    activation_checkpointing: Literal["none", "selective", "full"] = "selective"
    target_latent_tokens_phase1: int = 131_072
    target_latent_tokens_phase2: int = 65_536
    max_accumulation: int = 64
    overflow_policy: Literal["halve_tokens_lr_x0.7"] = "halve_tokens_lr_x0.7"


# ---------------------------------------------------------------------------
# inference (plan §15.5, §12.7)
# ---------------------------------------------------------------------------
class InferenceSpec(_Frozen):
    default_mode: Literal["faithful", "quality"] = "faithful"
    steps_fair: int = 1
    steps_quality: int = 4
    tile_lq: int = 256
    tile_overlap_lq: int = 64
    tile_overlap_hr: int = 256
    tile_blend: Literal["cosine-feather"] = "cosine-feather"
    tile_padding: Literal["reflect"] = "reflect"
    rope_absolute_coordinates: bool = True


# ---------------------------------------------------------------------------
# VAE weights (plan §4, §17.3)
# ---------------------------------------------------------------------------
class VAESpec(_Frozen):
    path: str = ""
    expected_sha256: str = ""


# ---------------------------------------------------------------------------
# data service (plan §10) + degradation (plan §11)
# ---------------------------------------------------------------------------
class DataSpec(_Frozen):
    """Raw source + index outputs (plan §10.1-§10.3, config/data.toml [data])."""

    raw_source: str = "danbooru-webdataset"
    manifest_dir: str = "data/index"
    outputs: list[str] = Field(
        default_factory=lambda: [
            "sr-eligibility-v1.parquet",
            "shard-summary-v1.parquet",
            "filter-report-v1.json",
            "sr-validation-v1.json",
        ]
    )
    # §11.4: offline real-codec LQ bank (None = disabled, synthetic chain only)
    bank_dir: str | None = None


class FilterSpec(_Frozen):
    """§10.4 filter funnel. Hard excludes are quality-gated: a fraction of
    hard-excluded samples is sampled for human review before final drop."""

    hard_exclude: list[str] = Field(
        default_factory=lambda: [
            "nsfw", "gore", "blood", "logo", "watermark", "signature", "text-heavy-ui"
        ]
    )
    human_review_fraction: float = 0.35
    priority_quality: list[str] = Field(default_factory=lambda: ["masterpiece", "best", "great", "good"])
    aux_tags: list[str] = Field(default_factory=lambda: ["monochrome", "rough", "3d"])
    aux_max_fraction: float = 0.20
    crop_retention_min: float = 0.80
    clean_score_stage: Literal["lazy"] = "lazy"
    clean_score_cache: bool = True


class ValidationSpec(_Frozen):
    """§16.1 validation sets; train/validation zero overlap is structural."""

    synthetic_pairs: int = 5000
    stress_set: int = 500
    real_lq_min: int = 200
    real_lq_target: int = 500
    zero_overlap: bool = True
    # deterministic split: sample id is validation iff
    # blake2b(sample_id) % 10000 < validation_permille.
    validation_permille: int = 300


class DegradationSpec(_Frozen):
    """§11 degradation contract (ranges + weights mirror config/data.toml)."""

    profiles: dict[str, float] = Field(
        default_factory=lambda: {
            "P0_clean": 0.10,
            "P1_mild_web": 0.25,
            "P2_normal_web": 0.35,
            "P3_anime_codec": 0.20,
            "P4_severe": 0.10,
        }
    )
    blur_sigma: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "mild": [0.1, 0.6],
            "normal": [0.2, 1.2],
            "severe": [0.5, 2.0],
        }
    )
    gaussian_noise: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "mild": [0.0, 2.0],
            "normal": [0.0, 5.0],
            "severe": [2.0, 10.0],
        }
    )
    jpeg_quality: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "mild": [80.0, 98.0],
            "normal": [55.0, 90.0],
            "severe": [30.0, 70.0],
        }
    )
    codec_bank_hr_crops_min: int = 50_000
    codec_bank_hr_crops_max: int = 100_000
    codec_bank_versions_per_crop: list[int] = Field(default_factory=lambda: [1, 2])
    codec_bank_batch_fraction: list[float] = Field(default_factory=lambda: [0.10, 0.20])
    # the fraction of the training batch whose LQ comes from the codec bank
    # (one value inside codec_bank_batch_fraction); 0.0 = synthetic chain only
    codec_bank_fraction: float = 0.15
    # §11.5: seed = H(global_seed, sample_id, data_cycle, exposure_index)
    seed: str = "H(global_seed, sample_id, data_cycle, exposure_index)"
    output_size: str = "hr/4 exact"


# ---------------------------------------------------------------------------
# M2 pixel baseline run (plan §M2, step 7)
# ---------------------------------------------------------------------------
class PixelBaselineSpec(_Frozen):
    """Small-scale pixel-baseline pilot (plan §M2: fidelity floor, not final quality)."""

    iterations: int = 100_000
    batch_size: int = 8
    base_channels: int = 160  # plan §M2 band 5M-10M (160/2 -> ~9.5M)
    depth: int = 2
    l1_weight: float = 1.0
    l2_weight: float = 0.0
    save_every_steps: int = 1_000
    val_every_steps: int = 5_000
    num_workers: int = 4
    out_dir: str = "output_model/pixel-baseline"


# ---------------------------------------------------------------------------
# M3/M4 latent flow run (plan §5, §13, §14; §4.3 pre-encode design)
# ---------------------------------------------------------------------------
class LatentFlowSpec(_Frozen):
    """Latent flow-matching loop over a pre-encoded z_hr store.

    z_hr is read from the LatentStore (fp16, crop pinned at box (0,0));
    z_lr = E_Mage(Bicubic4x(LQ)) is computed on the fly by the frozen VAE
    (plan §4.3). The exposure budget comes from [phase1] (the smoke overlay
    shrinks it to the M3 100k-200k window); the loop mechanics live here."""

    batch_size: int = 8
    save_every_steps: int = 1_000
    val_every_steps: int = 5_000
    val_samples: int = 8
    # double-buffered CPU prefetch: the next step's decode+crop+degrade batch
    # is fetched while the current step computes (M1 #8: data-wait gate)
    prefetch: bool = True
    out_dir: str = "output_model/latent-flow"


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------
class Config(_Frozen):
    project: ProjectSpec = Field(default_factory=ProjectSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    flow: FlowSpec = Field(default_factory=FlowSpec)
    buckets: BucketsSpec = Field(default_factory=BucketsSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    filter: FilterSpec = Field(default_factory=FilterSpec)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    degradation: DegradationSpec = Field(default_factory=DegradationSpec)
    phase1: Phase1Spec = Field(default_factory=Phase1Spec)
    phase2: Phase2Spec = Field(default_factory=Phase2Spec)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)
    gradient: GradientSpec = Field(default_factory=GradientSpec)
    ema: EMASpec = Field(default_factory=EMASpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    inference: InferenceSpec = Field(default_factory=InferenceSpec)
    vae: VAESpec = Field(default_factory=VAESpec)
    pixel_baseline: PixelBaselineSpec = Field(default_factory=PixelBaselineSpec)
    latent_flow: LatentFlowSpec = Field(default_factory=LatentFlowSpec)

    def validate_all(self) -> None:
        self.model.validate_structure()
        frac = sum(p.fraction for p in self.phase1.curriculum)
        if abs(frac - 1.0) > 1e-9:
            raise ValueError(f"Phase-I curriculum fractions must sum to 1.0, got {frac}")
        for p in self.phase1.curriculum:
            mix_sum = sum(p.mix.values())
            if abs(mix_sum - 1.0) > 1e-9:
                raise ValueError(f"curriculum {p.label}: mix must sum to 1.0, got {mix_sum}")
