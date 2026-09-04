"""Phase 5 spec-18 lambda=0 one-update parity chain (shared).

The single source of truth for the S18 chain, shared by:

* ``test_irepa_zero_lambda_optimizer_parity.py`` (single-rank HCU parity
  gate, pytest), and
* ``irepa_ddp_lambda_zero_smoke.py`` (2-rank DDP smoke, torchrun).

The chain: one no-iREPA source checkpoint N (production-shape composite,
two real updates, RAW checkpoint, saved by rank 0), then:

* Arm A (legacy / no-iREPA): continues from update N with one update N+1.
* Arm B (migrated iREPA): real ``migrate_irepa_checkpoint`` of the same
  source checkpoint (rank 0), production resume via
  ``load_raw_checkpoint`` into a fresh v4 composite + the PRODUCTION
  optimizer class (``hybrid_cmuon_canonical_ns4_fp32_rescue``, built from
  the real ``train_g1_fp32_rescue_r1.toml`` — guard config + per-(FQN,chunk)
  bootstrap references included), one update N+1 at
  ``lambda(N+1) == exact zero`` with the FULL teacher/projector/cosine
  graph still running (no-skip contract).

Controls (identical for both arms): exact same batch, timestep, noise,
(no dropout: all production dropout rates are 0), train RNG (no forward
RNG consumed), optimizer state (Arm B resumes the saved state bit-exact),
and optimizer SR RNG (saved and restored in the checkpoint).

In multi-rank mode every rank executes the identical collective sequence
(the guarded canonical optimizer performs its owner election, NS
broadcast, flag and rescue collectives internally; all are guarded by
``world_size > 1``).  Rank 0 alone writes the source checkpoint and runs
the migration; all ranks load the shared migrated checkpoint.  The
guard state is world-size stamped, so a 1-rank checkpoint cannot load
into a 2-rank optimizer (fail-closed by design) — the whole chain runs
at the target world size.

HCU determinism facts (measured on this backend, DTK torch 2.9.0+das,
salt13 2x BW): bf16 ``A@B`` matmul, bf16 reductions and the DiT
dense_sdpa forward/backward are bit-deterministic; bf16
``torch.addmm`` (the fused GEMM the BF16 Newton-Schulz iteration uses)
is NON-deterministic across calls for identical inputs on every
production chunk shape (``use_deterministic_algorithms`` does not fix
it); the FP32 NS path is bit-deterministic on every production chunk
shape.  The ``deterministic_ns`` flag swaps the NS entry point to the
deterministic FP32-NS computation with a single BF16 rounding at the
update boundary (exactly the production rescue staging, test-only
monkeypatch — production code untouched), making the primary gate fully
bit exact in both single-rank and 2-rank mode.

Model: the production composite (d=2560, depth 20, 20Q/5KV, head_dim
128, BF16) with the production TextConditioner and ConditionTokenEncoder
— a small model must not fake this parity (spec 20), and the production
``text.*`` decay FQNs sort AFTER ``irepa_alignment.*``: this is exactly
the SR-consumption ordering hazard that spec 11 (AdamW SR RNG audit)
requires the migration to keep out of the old-parameter stream
(projector appended after every existing AdamW parameter, never
interleaved).
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, cast

import torch
import torch.distributed as dist
from torch import nn

import sakuramoon.optim.fp32_rescue as fp32_rescue_module
import sakuramoon.optim.guarded_canonical as guarded_canonical_module
from sakuramoon.assets.pe_spatial import require_local_pe_spatial_teacher
from sakuramoon.checkpoint.load import load_raw_checkpoint
from sakuramoon.checkpoint.migrate_irepa_checkpoint import migrate_irepa_checkpoint
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.config import load_config
from sakuramoon.config.schema import IRepaConfig
from sakuramoon.encoders.pe_spatial import (
    FrozenPESpatialEncoder,
    prepare_teacher_targets,
)
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.model.irepa import IRepaAlignment
from sakuramoon.objective.flow import (
    flow_matching_loss,
    interpolate_state,
    sample_noise,
)
from sakuramoon.objective.irepa import (
    IRepaLambdaSchedule,
    irepa_alignment_loss,
    spatial_zscore_target,
)
from sakuramoon.optim.cmuon import cmuon_zeroth_power_fp32
from sakuramoon.optim.fp32_rescue import build_fp32_rescue
from sakuramoon.optim.guarded_canonical import GuardedCanonicalGuardConfig
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    TrainableComposite,
    TrainableCompositeInputs,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
TEACHER_DIR = "model/pe_spatial_b16_512"
PRODUCTION_OPTIMIZER_CONFIG = "train_g1_fp32_rescue_r1.toml"
PRODUCTION_OPTIMIZER_NAME = "hybrid_cmuon_canonical_ns4_fp32_rescue"

# Source checkpoint update N (spec 18) and the controlled batch/model seeds.
SOURCE_UPDATE = 2
NEXT_UPDATE = SOURCE_UPDATE + 1
MODEL_SEED = 20260905
BATCH_SEED = 90210
RGB_SEED = 31337
SR_SEED = 777
TAP_SLOT = 8
T = 0.37
T_EPS = 0.05
NOISE_OBSERVATION_BOUNDARY = 0.95

HIDDEN = 2560
LATENT_CHANNELS = 128
GRID = 8
QWEN_LEN = 16
MAIN_TOKENS = 3
CONDITION_TOKENS = 8
RESOLVED_CONFIG = b"[checkpoint]\nfull_every_updates = 100\n"
PROJECTOR_WEIGHT_FQN = "irepa_alignment.projector.weight"
PROJECTOR_BIAS_FQN = "irepa_alignment.projector.bias"

# Spec-17 HCU-leg tolerance for the NS-affected CMuon parameter values in
# the unpatched production-NS run: the cross-call BF16 addmm
# non-determinism is ulp-level, and the documented worst-case cross-NS-path
# difference (BF16 vs FP32 rescue) is delta-rms ratio p50 0.983 (~1.7%).
# 5e-2 relative RMS is ~30x that and still orders of magnitude below any
# checkpoint/state error (O(1) relative divergence).
PRODUCTION_NS_REL_RMS_TOLERANCE = 5e-2

_SECRET_ENVIRONMENT = {
    "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
    "WANDB_API_KEY": "synthetic-wandb-secret",
}

# The REAL production optimizer config (extends chain resolved by the
# config system): guard calibration values + the per-(FQN,chunk) bootstrap
# reference table + the canonical per-role NS map, exactly as G1 deploys
# them.  Loading it here (never hand-copying) makes the test drift-fail
# loudly if the production optimizer config changes.
_PRODUCTION_LOADED = load_config(
    Path(PRODUCTION_OPTIMIZER_CONFIG),
    config_root=REPOSITORY_ROOT / "config",
    environment=_SECRET_ENVIRONMENT,
)
_PRODUCTION_OPTIMIZER_CONFIG_OBJECT = _PRODUCTION_LOADED.config.optimizer
assert _PRODUCTION_OPTIMIZER_CONFIG_OBJECT.name == PRODUCTION_OPTIMIZER_NAME


def teacher_asset_available() -> bool:
    try:
        require_local_pe_spatial_teacher(REPOSITORY_ROOT, TEACHER_DIR)
    except Exception:  # noqa: BLE001 - any absence reason
        return False
    return True


def _deterministic_ns_bf16(
    grad: torch.Tensor,
    ns_steps: int,
    ns_coefficients: tuple[float, float, float],
    eps: float,
) -> torch.Tensor:
    """Deterministic stand-in for the production BF16 Newton-Schulz.

    Runs the bit-deterministic FP32 NS (same algorithm, coefficients,
    steps and normalization as ``cmuon_zeroth_power_bf16``; measured
    bit-identical across repeated calls on every production chunk shape
    on this HCU) and applies the single BF16 rounding at the update
    boundary — exactly how the production FP32 rescue stages its result.
    Test-only: production code is untouched.
    """
    return cmuon_zeroth_power_fp32(grad, ns_steps, ns_coefficients, eps).bfloat16()


class DeterministicNs:
    """Swap the BF16 NS entry point to the deterministic computation for
    the duration of one S18 chain.

    The production step resolves ``cmuon_zeroth_power_bf16`` through the
    importing module's namespace, so both consumer namespaces are patched
    (and restored, even on error) explicitly.
    """

    def __init__(self) -> None:
        self._patched: list[tuple[object, object]] = []

    def __enter__(self) -> Self:
        for module in (fp32_rescue_module, guarded_canonical_module):
            original = getattr(module, "cmuon_zeroth_power_bf16", None)
            if original is not None:
                module.cmuon_zeroth_power_bf16 = _deterministic_ns_bf16  # type: ignore[attr-defined]
                self._patched.append((module, original))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for module, original in self._patched:
            module.cmuon_zeroth_power_bf16 = original  # type: ignore[attr-defined]
        self._patched = []


def _production_composite(
    v4: bool, seed: int, device: torch.device
) -> TrainableComposite:
    """The production-shape composite, seeded so both arms start identical.

    The v4 arm constructs the projector LAST (after dit/text/condition), so
    the shared seed produces bit-identical old parameters in both arms.
    """

    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    dit = DenseDiT(
        depth=20,
        input_channels=LATENT_CHANNELS,
        hidden_size=HIDDEN,
        intermediate_size=6912,
        q_heads=20,
        kv_heads=5,
        head_dim=128,
        rope_nope_dim=32,
        rope_y_dim=48,
        rope_x_dim=48,
        rope_position_scale=16.0,
        rope_theta=1000.0,
        norm_eps=1e-6,
        timestep_dim=256,
        size_dim=64,
        aspect_dim=64,
        condition_hidden_size=1024,
        stable_slot_count=24,
        modulation_chunks=6,
        final_modulation_size=5120,
        out_channels=LATENT_CHANNELS,
        condition_token_count=CONDITION_TOKENS,
        modality_init_std=0.02,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    ).to(device=device)  # type: ignore[arg-type]
    text = TextConditioner.for_production(
        attention_heads=16,
        mix_gate_init=0.0,
        layer_scale_init=1.0,
        projection_bias=False,
    ).to(device=device)
    condition = ConditionTokenEncoder(
        input_size=2048,
        hidden_size=1024,
        intermediate_size=2048,
        output_size=HIDDEN,
        token_count=CONDITION_TOKENS,
        attention_heads=16,
        norm_eps=1e-6,
        init_std=0.02,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    ).to(device=device)
    irepa: IRepaAlignment | None = None
    if v4:
        irepa = IRepaAlignment(HIDDEN).to(device=device)
    return TrainableComposite(
        dit=dit,
        text=text,
        condition_tokens=condition,
        irepa_alignment=irepa,
        irepa_tap_slot_id=TAP_SLOT if v4 else None,
    )


def _production_optimizer(
    module: TrainableComposite, *, rank: int, world_size: int
):
    """The PRODUCTION optimizer class, built from the production config.

    ``rank``/``world_size`` select the canonical owner mapping; in a
    single-process run (world size 1, rank 0) every cross-rank collective
    in the guarded canonical flow is guarded by ``world_size > 1`` and
    stays off, so no ``torch.distributed`` init is needed.  In 2-rank mode
    the caller must have initialized the process group first.  ``lr`` is
    the config's pre-batch-scaling ``base_lr`` (both arms identical; the
    guard band is proportional to ``lr`` so the safety semantics are
    unchanged).
    """
    optimizer = _PRODUCTION_OPTIMIZER_CONFIG_OBJECT
    guard = optimizer.cmuon_guard
    assert guard is not None and guard.enabled
    assert optimizer.cmuon_ns is not None
    return build_fp32_rescue(
        module,
        lr=optimizer.base_lr,
        betas=(float(optimizer.betas[0]), float(optimizer.betas[1])),
        eps=optimizer.eps,
        block_size=optimizer.block_size,
        bf16_stochastic_round=optimizer.bf16_stochastic_round,
        matrix_weight_decay=optimizer.matrix_weight_decay,
        sensitive_weight_decay=optimizer.sensitive_weight_decay,
        sr_seed=SR_SEED,
        ns_steps_by_role=optimizer.cmuon_ns.canonical_map(),
        guard_cfg=GuardedCanonicalGuardConfig(
            guard_ratio=guard.guard_ratio,
            reference_decay=guard.reference_decay,
            min_reference=guard.min_reference,
            numerical_floor=guard.numerical_floor,
            warmup_observations=guard.warmup_observations,
            invariant_check=guard.invariant_check,
        ),
        guard_bootstrap_refs=dict(guard.references),
        rank=rank,
        world_size=world_size,
        momentum_dtype=optimizer.cmuon_momentum_dtype,
        chunk_rescale_sqrt_n=optimizer.cmuon_chunk_rescale_sqrt_n,
    )


def _batch_inputs(
    seed: int,
    device: torch.device,
) -> tuple[TrainableCompositeInputs, torch.Tensor, torch.Tensor]:
    """One deterministic flow-matching batch (inputs, state, clean)."""

    generator = torch.Generator(device=device).manual_seed(seed)
    qwen_states = torch.randn(
        1, QWEN_LEN, 7, 2048, generator=generator, device=device
    ).to(torch.bfloat16)
    main_token_indices = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    main_mask = torch.ones(1, MAIN_TOKENS, dtype=torch.bool, device=device)
    condition_token_indices = torch.tensor(
        [[3 + index for index in range(CONDITION_TOKENS)]],
        dtype=torch.long,
        device=device,
    )
    condition_mask = torch.ones(1, CONDITION_TOKENS, dtype=torch.bool, device=device)
    use_null_condition = torch.zeros(1, dtype=torch.bool, device=device)
    active_condition_sample_indices = torch.zeros(1, dtype=torch.long, device=device)
    clean = torch.randn(
        1, LATENT_CHANNELS, GRID, GRID, generator=generator, device=device
    ).to(torch.bfloat16)
    timestep = torch.full((1,), T, dtype=torch.float32, device=device)
    noise = sample_noise(clean, noise_scale=1.0, generator=generator)
    state = interpolate_state(clean, noise, timestep)
    inputs = TrainableCompositeInputs(
        qwen_states=qwen_states,
        main_token_indices=main_token_indices,
        main_mask=main_mask,
        main_token_lengths=(MAIN_TOKENS,),
        condition_token_indices=condition_token_indices,
        condition_mask=condition_mask,
        use_null_condition=use_null_condition,
        active_condition_sample_indices=active_condition_sample_indices,
        # Per-sample [C,H,W] / [H*W,2]: forward_dit stacks into [B,...].
        latents=(state[0],),
        image_coordinates=(image_coordinates(GRID, GRID, device=device),),
        timestep=timestep,
        size_scale=torch.zeros(1, dtype=torch.float32, device=device),
        aspect=torch.zeros(1, dtype=torch.float32, device=device),
        growth_alpha=1.0,
    )
    return inputs, state, clean


def _main_per_sample(
    predictions: tuple[torch.Tensor, ...],
    inputs: TrainableCompositeInputs,
    state: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    prediction_batch = torch.stack(predictions)
    return flow_matching_loss(
        prediction_batch,
        state,
        clean,
        inputs.timestep,
        t_eps=T_EPS,
        noise_observation_boundary=NOISE_OBSERVATION_BOUNDARY,
    ).per_sample


def _irepa_per_sample(
    projected: torch.Tensor,
    teacher: FrozenPESpatialEncoder,
    device: torch.device,
) -> torch.Tensor:
    """The full teacher/projector/cosine graph (frozen teacher, no grad)."""

    generator = torch.Generator(device=device).manual_seed(RGB_SEED)
    rgb = (
        2.0
        * torch.randn(1, 3, 16 * GRID, 16 * GRID, generator=generator, device=device)
        - 1.0
    ).to(torch.bfloat16)
    with torch.no_grad():
        teacher_output = prepare_teacher_targets(teacher, rgb)
    target = spatial_zscore_target(teacher_output.patch_features, gamma=1.0, eps=1e-5)
    return irepa_alignment_loss(projected, target).per_sample


def _raw_state(update: int) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=SingleGpuUpdateState(update, update, update),
        growth=GrowthCheckpointState(
            active_slot_ids(20), 1.0, "stage1", 1, 1024, None, None
        ),
        stage_budget=StageBudgetCheckpointState(0, 1000),
        checkpoint_cadence=CheckpointCadence(update, float(update), 100),
    )


def _irepa_config() -> IRepaConfig:
    return IRepaConfig(
        enabled=True,
        teacher_id="facebook/PE-Spatial-B16-512",
        tap_slot=TAP_SLOT,
        projector_kernel_size=3,
        spatial_norm="zscore",
        loss="cosine",
    )


def _adamw_state_by_name(
    outer_state: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Map the inner AdamW8bit state_dict from parameter id to FQN."""

    inner = cast("dict[str, object]", outer_state["optimizer"])
    state = cast("dict[int, dict[str, object]]", inner["state"])
    groups = cast("list[dict[str, object]]", inner["param_groups"])
    by_name: dict[str, dict[str, object]] = {}
    for group in groups:
        params = cast(list[int], group["params"])
        names = cast(list[str], group["param_names"])
        for param_id, name in zip(params, names, strict=True):
            entry = state[param_id]
            by_name[name] = dict(entry)
    return by_name


def _guard_state(outer_state: Mapping[str, object]) -> Mapping[str, object]:
    guard = outer_state["guard"]
    assert isinstance(guard, Mapping)
    return cast(Mapping[str, object], guard)


def _sr_rng_state(outer_state: Mapping[str, object]) -> torch.Tensor:
    """The optimizer's shared stochastic-rounding RNG state (the resume control)."""

    sr = cast("Mapping[str, object]", outer_state["sr_rng"])
    return cast(torch.Tensor, sr["state"])


def _step_int(entry: Mapping[str, object]) -> int:
    return int(cast(torch.Tensor, entry["step"]).item())


def _fp32_rescue_block(guard: Mapping[str, object]) -> dict[str, object]:
    return cast(dict[str, object], guard.get("fp32_rescue"))


def _counter(block: Mapping[str, object], key: str) -> int:
    return int(cast(int, block[key]))


class _QuantizedMoment(Protocol):
    """The stored torchao OptimState8bit representation (duck-typed)."""

    block_size: int
    codes: torch.Tensor
    qmap: torch.Tensor
    scale: torch.Tensor


def _moment_exact(a: object, b: object) -> bool:
    """Bit-exact comparison of one AdamW8bit moment entry.

    BF16-decay parameters carry torchao ``OptimState8bit`` quantized state,
    which does not implement ``aten.equal``; the exact comparison is on the
    stored representation (block_size + codes/qmap/scale).  Sensitive
    parameters carry plain FP32 tensors and compare directly.
    """
    if type(a).__name__ == "OptimState8bit" or type(b).__name__ == "OptimState8bit":
        if type(a) is not type(b):
            return False
        a_q = cast(_QuantizedMoment, a)
        b_q = cast(_QuantizedMoment, b)
        return (
            a_q.block_size == b_q.block_size
            and torch.equal(a_q.codes, b_q.codes)
            and torch.equal(a_q.qmap, b_q.qmap)
            and torch.equal(a_q.scale, b_q.scale)
        )
    return torch.equal(cast(torch.Tensor, a), cast(torch.Tensor, b))


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative RMS divergence of two same-shaped tensors."""

    diff = a.float() - b.float()
    scale = a.float().pow(2).mean().sqrt() + 1e-12
    return float((diff.pow(2).mean().sqrt() / scale).item())


@dataclass
class ChainResult:
    """Everything the two gates compare after the one-update chain."""

    main_a: torch.Tensor
    main_b: torch.Tensor
    total_b: torch.Tensor
    mean_loss_a: torch.Tensor
    mean_loss_b: torch.Tensor
    params_a: dict[str, nn.Parameter]
    params_b: dict[str, nn.Parameter]
    successful_updates_a: int
    successful_updates_b: int
    attempted_updates_a: int
    attempted_updates_b: int
    sr_equal_at_resume: bool
    cmuon_momenta_a: Mapping[str, torch.Tensor]
    cmuon_momenta_b: Mapping[str, torch.Tensor]
    adamw_a: dict[str, dict[str, object]]
    adamw_b: dict[str, dict[str, object]]
    guard_a: Mapping[str, object]
    guard_b: Mapping[str, object]


def run_chain(
    tmp_path: Path,
    *,
    deterministic_ns: bool,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> ChainResult:
    """Run the full S18 chain on one rank (both ranks call with the same
    arguments in multi-rank mode; rank 0 owns the checkpoint writes)."""

    context = DeterministicNs() if deterministic_ns else nullcontext()
    with context:
        # -- source checkpoint N (no-iREPA), two real updates -----------------
        arm_a = _production_composite(v4=False, seed=MODEL_SEED, device=device)
        optimizer_a = _production_optimizer(
            arm_a, rank=rank, world_size=world_size
        )
        inputs, state, clean = _batch_inputs(BATCH_SEED, device)
        step_a = SingleGpuStep(
            arm_a,
            optimizer_a,
            accumulation_steps=1,
            state=SingleGpuUpdateState.initial(),
        )
        for _ in range(SOURCE_UPDATE):
            source_output = arm_a.forward(inputs)
            assert isinstance(source_output, tuple)
            per_sample = _main_per_sample(source_output, inputs, state, clean)
            step_a.backward(per_sample)
            step_a.finish_update()
        assert optimizer_a.adamw.audit_state()[0].step == SOURCE_UPDATE

        if rank == 0:
            source_path = save_raw_checkpoint(
                tmp_path / "source-root",
                CheckpointIdentity("raw-parity-source-update-cadence", SOURCE_UPDATE),
                arm_a,
                optimizer_a,
                _raw_state(SOURCE_UPDATE),
                resolved_config=RESOLVED_CONFIG,
            ).path
            assert source_path.exists()

            # -- Arm B: migrate the SAME source checkpoint, verify exact ------
            # Production config: both weight decays are 0.0
            # (train_g1_fp32_rescue_r1 extends the s0 base with
            # matrix/sensitive_weight_decay = 0.0).
            migrated = migrate_irepa_checkpoint(
                source_path,
                tmp_path / "migrated",
                irepa=_irepa_config(),
                matrix_weight_decay=0.0,
                sensitive_weight_decay=0.0,
                migration_seed=MODEL_SEED,
            )
            assert isinstance(migrated, Path)
        if world_size > 1:
            dist.barrier()  # pyright: ignore[reportUnknownMemberType]
        migrated = tmp_path / "migrated"

        schedule = IRepaLambdaSchedule(
            start_successful_update=NEXT_UPDATE,
            target_weight=1.0,
            ramp_in_updates=1000,
            ramp_out_after_updates=None,
            ramp_out_updates=1000,
        )
        lambda_weight = schedule.weight_for_update(NEXT_UPDATE)
        assert lambda_weight == 0.0

        # Resume-time SR RNG control: the migrated checkpoint must restore
        # the exact optimizer SR RNG state the source arm had at update N.
        sr_a_at_n = _sr_rng_state(optimizer_a.state_dict())

        arm_b = _production_composite(v4=True, seed=MODEL_SEED, device=device)
        optimizer_b = _production_optimizer(
            arm_b, rank=rank, world_size=world_size
        )
        restored = load_raw_checkpoint(
            migrated,
            arm_b,
            optimizer_b,
            CheckpointIdentity("raw-parity-source-update-cadence", SOURCE_UPDATE),
        )
        assert restored.trainer.successful_updates == SOURCE_UPDATE
        sr_b_at_n = _sr_rng_state(optimizer_b.state_dict())
        sr_equal_at_resume = bool(torch.equal(sr_a_at_n, sr_b_at_n))

        # State-exact resume: every pre-existing parameter already matches.
        params_a = dict(arm_a.named_parameters())
        params_b = dict(arm_b.named_parameters())
        assert set(params_a) < set(params_b)
        for name, parameter in params_a.items():
            assert torch.equal(parameter, params_b[name]), (
                f"pre-existing parameter differs after resume: {name}"
            )

        step_b = SingleGpuStep(
            arm_b,
            optimizer_b,
            accumulation_steps=1,
            state=SingleGpuUpdateState(SOURCE_UPDATE, SOURCE_UPDATE, SOURCE_UPDATE),
            irepa_projector=True,
        )

        # -- Arm A: the legacy no-iREPA reference update N+1 -------------------
        output_a = arm_a.forward(inputs)
        assert isinstance(output_a, tuple)
        main_a = _main_per_sample(output_a, inputs, state, clean)
        step_a.backward(main_a)
        result_a = step_a.finish_update()

        # -- Arm B: one update at lambda=0 with the full iREPA graph -----------
        output_b = arm_b.forward(inputs)
        assert not isinstance(output_b, tuple)
        main_b = _main_per_sample(output_b.predictions, inputs, state, clean)
        irepa_b = _irepa_per_sample(
            output_b.projected_student_features,
            FrozenPESpatialEncoder.load_asset(REPOSITORY_ROOT, TEACHER_DIR, device=device),
            device,
        )
        assert torch.isfinite(irepa_b).all()
        # No-skip contract: the weighted term stays in the graph at lambda=0.
        total_b = main_b + lambda_weight * irepa_b
        step_b.backward(total_b)
        result_b = step_b.finish_update()

        outer_a = optimizer_a.state_dict()
        outer_b = optimizer_b.state_dict()
        cmuon_a = cast("Mapping[str, object]", outer_a["cmuon"])
        cmuon_b = cast("Mapping[str, object]", outer_b["cmuon"])
        momenta_a = cast(Mapping[str, torch.Tensor], cmuon_a["momenta"])
        momenta_b = cast(Mapping[str, torch.Tensor], cmuon_b["momenta"])

    return ChainResult(
        main_a=main_a,
        main_b=main_b,
        total_b=total_b,
        mean_loss_a=result_a.mean_loss,
        mean_loss_b=result_b.mean_loss,
        params_a=params_a,
        params_b=params_b,
        successful_updates_a=step_a.state.successful_updates,
        successful_updates_b=step_b.state.successful_updates,
        attempted_updates_a=step_a.state.attempted_updates,
        attempted_updates_b=step_b.state.attempted_updates,
        sr_equal_at_resume=sr_equal_at_resume,
        cmuon_momenta_a=momenta_a,
        cmuon_momenta_b=momenta_b,
        adamw_a=_adamw_state_by_name(outer_a),
        adamw_b=_adamw_state_by_name(outer_b),
        guard_a=_guard_state(outer_a),
        guard_b=_guard_state(outer_b),
    )


def assert_deterministic_parts(result: ChainResult) -> None:
    """The comparisons that must be bit exact in BOTH gates."""

    # MAIN JLT loss bit exact; TOTAL == MAIN bit exact (lambda=0).
    assert torch.equal(result.main_a, result.main_b), (
        "MAIN JLT per-sample loss is not bit exact"
    )
    assert torch.equal(result.total_b, result.main_b), (
        "lambda=0 TOTAL loss is not MAIN bit exact"
    )
    assert torch.equal(result.mean_loss_a, result.mean_loss_b)

    # The spec-18 resume control: exact same optimizer SR RNG at update N.
    assert result.sr_equal_at_resume, "optimizer SR RNG not restored at resume"

    # Every pre-existing NON-CMuon parameter (the AdamW part — including the
    # SR-consumption ordering claim: the projector's draw is last, so no old
    # AdamW draw is shifted): bit exact in both gates.  The CMuon parameters
    # are checked by the mode-specific function (bit exact in the primary
    # gate, tolerance in the spec-17 HCU leg).
    cmuon_names = set(result.cmuon_momenta_a)
    for name, parameter in result.params_a.items():
        if name in cmuon_names:
            continue
        assert torch.equal(parameter, result.params_b[name]), (
            f"pre-existing non-CMuon parameter diverged: {name}"
        )

    # AdamW state (step + both moments) for every pre-existing parameter.
    for name in result.adamw_a:
        assert name in result.adamw_b, f"pre-existing AdamW parameter missing: {name}"
        entry_a = result.adamw_a[name]
        entry_b = result.adamw_b[name]
        assert _step_int(entry_a) == _step_int(entry_b), name
        for field_name in ("exp_avg", "exp_avg_sq"):
            assert _moment_exact(
                result.adamw_a[name][field_name],
                result.adamw_b[name][field_name],
            ), f"AdamW {field_name} diverged: {name}"
    # The projector state exists only in Arm B (new, allowed to differ).
    assert PROJECTOR_WEIGHT_FQN in result.adamw_b
    assert PROJECTOR_BIAS_FQN in result.adamw_b
    assert PROJECTOR_WEIGHT_FQN not in result.adamw_a
    assert PROJECTOR_BIAS_FQN not in result.adamw_a

    # CMuon momenta (deterministic lerp — bit exact in both gates).
    assert set(result.cmuon_momenta_a) == set(result.cmuon_momenta_b)
    for name in result.cmuon_momenta_a:
        device_a = result.cmuon_momenta_a[name].device
        device_b = result.cmuon_momenta_b[name].device
        assert torch.equal(
            result.cmuon_momenta_a[name].to(device_a),
            result.cmuon_momenta_b[name].to(device_b),
        ), f"CMuon momentum diverged: {name}"

    # Guard references / bookkeeping: driven only by the deterministic
    # fp32 chunk signals, never by the NS output — bit exact in both gates.
    guard_a = result.guard_a
    guard_b = result.guard_b
    for key in (
        "schema_version",
        "config",
        "references",
        "skip_total",
        "skip_by_role",
        "skip_by_fqn",
        "observations",
        "bootstrap_mode",
        "owner_mapping_version",
        "world_size",
        "ns_map",
    ):
        assert key in guard_a and key in guard_b, f"guard state missing {key}"
        assert guard_a[key] == guard_b[key], f"guard state diverged: {key}"

    # Successful update counters.
    assert result.successful_updates_a == NEXT_UPDATE
    assert result.successful_updates_b == NEXT_UPDATE
    assert result.attempted_updates_a == NEXT_UPDATE
    assert result.attempted_updates_b == NEXT_UPDATE

    # The projector exists only in Arm B; after zero_grad its grad is gone.
    assert PROJECTOR_WEIGHT_FQN in result.params_b
    assert PROJECTOR_BIAS_FQN in result.params_b
    assert result.params_b[PROJECTOR_WEIGHT_FQN].grad is None
    assert result.params_b[PROJECTOR_BIAS_FQN].grad is None


def assert_primary_gate(result: ChainResult) -> None:
    """PRIMARY gate (spec 18): every pre-existing model parameter bit exact
    (CMuon and AdamW alike) plus the deterministic rescue bookkeeping."""

    for name, parameter in result.params_a.items():
        assert torch.equal(parameter, result.params_b[name]), (
            f"pre-existing parameter diverged after one lambda=0 update: {name}"
        )

    # Rescue bookkeeping is deterministic under the deterministic NS: the
    # two arms make identical safety decisions (and no safety failure may
    # occur).
    rescue_a = _fp32_rescue_block(result.guard_a)
    rescue_b = _fp32_rescue_block(result.guard_b)
    for key in (
        "bf16_attempts",
        "bf16_safety_failures",
        "fp32_attempts",
        "fp32_rescues",
        "fp32_rescue_failures",
        "rescue_by_role",
    ):
        assert rescue_a[key] == rescue_b[key], f"fp32_rescue state diverged: {key}"
    assert _counter(rescue_a, "fp32_rescue_failures") == 0
    assert _counter(rescue_a, "bf16_safety_failures") == 0


def guard_rescue_snapshot(guard: Mapping[str, object]) -> dict[str, object]:
    """JSON-serializable guard rescue bookkeeping snapshot (for reports).

    Reads the ``observations`` counter from the guard itself and the
    fp32_rescue attempt/rescue/failure counters from its rescue block.
    """
    block = _fp32_rescue_block(guard)
    return {
        "observations": _counter(guard, "observations"),
        "bf16_attempts": _counter(block, "bf16_attempts"),
        "bf16_safety_failures": _counter(block, "bf16_safety_failures"),
        "fp32_attempts": _counter(block, "fp32_attempts"),
        "fp32_rescues": _counter(block, "fp32_rescues"),
        "fp32_rescue_failures": _counter(block, "fp32_rescue_failures"),
        "rescue_by_role": block.get("rescue_by_role"),
    }


def production_ns_gate_report(result: ChainResult) -> dict[str, object]:
    """Spec-17 HCU leg: NS-affected CMuon parameter values within tolerance,
    no safety failure.  Returns the measured report (for logging)."""

    worst_name = ""
    worst_rel_rms = 0.0
    for name in result.cmuon_momenta_a:
        parameter_a = result.params_a[name]
        parameter_b = result.params_b[name]
        assert torch.isfinite(parameter_b.float()).all(), f"nonfinite param: {name}"
        rel_rms = _rel_rms(parameter_a, parameter_b)
        assert rel_rms <= PRODUCTION_NS_REL_RMS_TOLERANCE, (
            f"CMuon parameter {name} diverged beyond the HCU non-determinism "
            f"tolerance: rel-rms {rel_rms:.3e} > "
            f"{PRODUCTION_NS_REL_RMS_TOLERANCE:.3e} — not explainable by the "
            "BF16 addmm cross-call non-determinism; inspect the checkpoint "
            "path (spec 17: HCU nondeterminism must not mask a checkpoint "
            "error)"
        )
        if rel_rms > worst_rel_rms:
            worst_rel_rms = rel_rms
            worst_name = name

    # No safety failure may occur on the raw production path either; the
    # rescue counters may legitimately differ between the two independent
    # NS calls of the same update (rescue asymmetry), but neither arm may
    # have failed.
    rescue_a = _fp32_rescue_block(result.guard_a)
    rescue_b = _fp32_rescue_block(result.guard_b)
    assert _counter(rescue_a, "bf16_attempts") == _counter(rescue_b, "bf16_attempts")
    assert _counter(rescue_a, "fp32_rescue_failures") == 0
    assert _counter(rescue_b, "fp32_rescue_failures") == 0

    return {
        "worst_rel_rms": worst_rel_rms,
        "worst_param": worst_name,
        "rescue_a": dict(rescue_a),
        "rescue_b": dict(rescue_b),
    }
