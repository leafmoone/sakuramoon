"""Phase 4 iREPA training-graph GPU contract tests (DCU).

Covers the representation-alignment training graph on the real accelerators:

* the slot_08 stable-slot capture is EXACTLY the tapped block's output
  (verified against a test-only forward hook) and the image span only;
* the capture token count is the row-major image grid (``T == H * W``) for
  both square and rectangular shapes, at the production growth depths 20/24;
* the dense (SDPA) and packed (FA4 varlen) backends produce the same capture
  for identical weights and inputs;
* the capture is bit-stable across the activation-checkpoint modes
  (none/alternating/all) and numerically stable across torch.compile
  (eager vs compiled) — the capture point is an eager outer-loop read, never
  a computation inside a compiled or checkpointed block;
* the real frozen PE-Spatial teacher feeds the FP32 spatial z-score target,
  the projector, and the per-sample cosine loss, and the lambda-weighted
  objective backpropagates finite, nonzero gradients into the projector
  (weight and bias) on DCU;
* at lambda = 0 the iREPA contribution is an exact zero (SakuraMoon no-skip
  contract) even on the accelerator path: the projector grad is present as an
  exact-zero tensor, not absent.

The teacher-dependent tests require the fingerprint-bound PE-Spatial-B16-512
asset and skip only if it is absent from the checkout (a deployment gap, not a
regression).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sakuramoon.assets.pe_spatial import require_local_pe_spatial_teacher
from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.encoders.pe_spatial import (
    FrozenPESpatialEncoder,
    prepare_teacher_targets,
)
from sakuramoon.model.dit import DenseDiT, PackedDiT
from sakuramoon.model.growth import slot_name
from sakuramoon.model.irepa import IRepaAlignment
from sakuramoon.objective.irepa import (
    irepa_alignment_loss,
    spatial_zscore_target,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
DEVICE = torch.device("cuda", 0)
TAP_SLOT = 8
HIDDEN = 16
INPUT_CHANNELS = 8
CONDITION_TOKEN_COUNT = 8
TEXT_LEN = 3
TEACHER_DIR = "model/pe_spatial_b16_512"


def _teacher_asset_available() -> bool:
    try:
        require_local_pe_spatial_teacher(REPOSITORY_ROOT, TEACHER_DIR)
    except Exception:  # noqa: BLE001 - any absence reason skips the chain tests
        return False
    return True


TEACHER_AVAILABLE = _teacher_asset_available()
requires_teacher = pytest.mark.skipif(
    not TEACHER_AVAILABLE,
    reason="PE-Spatial-B16-512 teacher asset is not present in this checkout",
)


def _dit_kwargs(linear_dtype: torch.dtype) -> dict[str, object]:
    return {
        "input_channels": INPUT_CHANNELS,
        "hidden_size": HIDDEN,
        "intermediate_size": 32,
        "q_heads": 2,
        "kv_heads": 1,
        "head_dim": 8,
        "rope_nope_dim": 0,
        "rope_y_dim": 4,
        "rope_x_dim": 4,
        "rope_position_scale": 1.0,
        "rope_theta": 10.0,
        "norm_eps": 1e-6,
        "timestep_dim": 256,
        "size_dim": 64,
        "aspect_dim": 64,
        "condition_hidden_size": 1024,
        "stable_slot_count": 24,
        "modulation_chunks": 6,
        "final_modulation_size": 32,
        "out_channels": INPUT_CHANNELS,
        "condition_token_count": CONDITION_TOKEN_COUNT,
        "modality_init_std": 0.02,
        "linear_dtype": linear_dtype,
        "sensitive_dtype": torch.float32,
        "projection_bias": False,
        "attention_dropout": 0.0,
        "mlp_dropout": 0.0,
        "output_weight_zero_init": True,
        "output_bias_zero_init": True,
    }


def _small_dit(depth: int, *, linear_dtype: torch.dtype) -> DenseDiT:
    kwargs = _dit_kwargs(linear_dtype)
    kwargs["depth"] = depth
    # Move to the accelerator only: the parameters are already constructed in
    # their locked dtypes (linear in linear_dtype, sensitive in float32) and a
    # blanket .to(dtype=...) would corrupt the mixed-precision contract.
    return DenseDiT(**kwargs).to(device=DEVICE)  # type: ignore[arg-type]


def _forward_inputs(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
    seed: int = 17,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    batch = 1
    latent = torch.randn(
        batch, INPUT_CHANNELS, height, width, generator=generator, device=DEVICE
    ).to(dtype)
    text_tokens = torch.randn(
        batch, TEXT_LEN, HIDDEN, generator=generator, device=DEVICE
    ).to(dtype)
    text_mask = torch.ones(batch, TEXT_LEN, dtype=torch.bool, device=DEVICE)
    condition_tokens = torch.randn(
        batch, CONDITION_TOKEN_COUNT, HIDDEN, generator=generator, device=DEVICE
    ).to(dtype)
    condition_active_mask = torch.ones(batch, dtype=torch.bool, device=DEVICE)
    timestep = torch.full((batch,), 0.37, dtype=torch.float32, device=DEVICE)
    size_scale = torch.zeros(batch, dtype=torch.float32, device=DEVICE)
    aspect = torch.zeros(batch, dtype=torch.float32, device=DEVICE)
    image_coords = image_coordinates(height, width, device=DEVICE).unsqueeze(0)
    return {
        "latent": latent,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "condition_tokens": condition_tokens,
        "condition_active_mask": condition_active_mask,
        "timestep": timestep,
        "size_scale": size_scale,
        "aspect": aspect,
        "image_coordinates": image_coords,
    }


def _tapped_capture(dit: DenseDiT, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    _predictions, capture = dit.forward_tapped(
        inputs["latent"],
        inputs["text_tokens"],
        inputs["text_mask"],
        inputs["condition_tokens"],
        inputs["condition_active_mask"],
        inputs["timestep"],
        inputs["size_scale"],
        inputs["aspect"],
        image_coordinates=inputs["image_coordinates"],
        growth_alpha=1.0,
        tap_slot_id=TAP_SLOT,
    )
    return capture


def _seeded(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _remove_dynamo_compiled_artifacts() -> None:
    """Remove torch.dynamo ``__compiled_fn_*`` namespace artifacts.

    ``torch.compile`` (the compile-stability test below) traces through the
    attention module and injects compiled-frame wrappers (``__compiled_fn_*``)
    into the module namespaces it touches.  A later unit test
    (``test_distributed_compile.test_fa2_is_the_only_explicit_eager_compiler_
    boundary``) asserts the attention module's eager-boundary set, which such
    artifacts would pollute.  This keeps the GPU training-graph tests isolated
    from the rest of the session.
    """
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("sakuramoon") or mod is None:
            continue
        for name in list(vars(mod).keys()):
            if name.startswith("__compiled_fn_"):
                try:
                    delattr(mod, name)
                except (AttributeError, TypeError):
                    pass


@pytest.fixture(autouse=True)
def _keep_dynamo_artifacts_local():
    _remove_dynamo_compiled_artifacts()
    yield
    _remove_dynamo_compiled_artifacts()


@pytest.mark.parametrize(("height", "width"), [(4, 4), (3, 5)])
def test_slot08_capture_is_tapped_block_output(height: int, width: int) -> None:
    dit = _small_dit(20, linear_dtype=torch.float32)
    dit.eval()
    inputs = _forward_inputs(height, width, dtype=torch.float32)

    recorded: list[torch.Tensor] = []

    def _record(_module: nn.Module, _in: object, output: torch.Tensor) -> None:
        recorded.append(output.detach())

    handle = dit.blocks[slot_name(TAP_SLOT)].register_forward_hook(_record)
    try:
        capture = _tapped_capture(dit, inputs)
    finally:
        handle.remove()

    assert len(recorded) == 1
    block_output = recorded[0]
    image_start = TEXT_LEN + CONDITION_TOKEN_COUNT
    image_tokens = height * width
    want_span = block_output[:, image_start : image_start + image_tokens]
    assert tuple(capture.shape) == (1, image_tokens, HIDDEN)
    # The capture is EXACTLY the image span of the tapped block's output
    # (bitwise, not close): the tap reads the block output, nothing else.
    assert torch.equal(capture, want_span)


@pytest.mark.parametrize(("height", "width"), [(4, 4), (3, 5), (2, 8)])
@pytest.mark.parametrize("depth", [20, 24])
def test_slot08_capture_token_count_is_row_major_grid(
    height: int, width: int, depth: int
) -> None:
    dit = _small_dit(depth, linear_dtype=torch.float32)
    dit.eval()
    inputs = _forward_inputs(height, width, dtype=torch.float32)
    capture = _tapped_capture(dit, inputs)

    # T == H * W exactly (no pad-to-square, no resize).
    assert tuple(capture.shape) == (1, height * width, HIDDEN)
    assert bool(torch.isfinite(capture).all())


def _production_kwargs() -> dict[str, object]:
    # The PackedDiT FA4 varlen backend is locked to the production attention
    # config (d=2560, 20Q/5KV, head_dim=128), so the dense-vs-packed capture
    # parity is exercised with the full production model in bfloat16.  Depth
    # is 20 (not the base 16): the iREPA tap is stable slot 8, a G1 growth
    # slot that is only active at depth 20/24.
    return {
        "depth": 20,
        "input_channels": 128,
        "hidden_size": 2560,
        "intermediate_size": 6912,
        "q_heads": 20,
        "kv_heads": 5,
        "head_dim": 128,
        "rope_nope_dim": 32,
        "rope_y_dim": 48,
        "rope_x_dim": 48,
        "rope_position_scale": 16.0,
        "rope_theta": 1000.0,
        "norm_eps": 1e-6,
        "timestep_dim": 256,
        "size_dim": 64,
        "aspect_dim": 64,
        "condition_hidden_size": 1024,
        "stable_slot_count": 24,
        "modulation_chunks": 6,
        "final_modulation_size": 5120,
        "out_channels": 128,
        "condition_token_count": 8,
        "modality_init_std": 0.02,
        "linear_dtype": torch.bfloat16,
        "sensitive_dtype": torch.float32,
        "projection_bias": False,
        "attention_dropout": 0.0,
        "mlp_dropout": 0.0,
        "output_weight_zero_init": True,
        "output_bias_zero_init": True,
    }


def _production_forward_inputs(
    height: int, width: int, *, seed: int = 23
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    batch = 1
    hidden = 2560
    latent = torch.randn(
        batch, 128, height, width, generator=generator, device=DEVICE
    ).to(torch.bfloat16)
    text_tokens = torch.randn(
        batch, TEXT_LEN, hidden, generator=generator, device=DEVICE
    ).to(torch.bfloat16)
    text_mask = torch.ones(batch, TEXT_LEN, dtype=torch.bool, device=DEVICE)
    condition_tokens = torch.randn(
        batch, CONDITION_TOKEN_COUNT, hidden, generator=generator, device=DEVICE
    ).to(torch.bfloat16)
    condition_active_mask = torch.ones(batch, dtype=torch.bool, device=DEVICE)
    timestep = torch.full((batch,), 0.37, dtype=torch.float32, device=DEVICE)
    size_scale = torch.zeros(batch, dtype=torch.float32, device=DEVICE)
    aspect = torch.zeros(batch, dtype=torch.float32, device=DEVICE)
    image_coords = image_coordinates(height, width, device=DEVICE).unsqueeze(0)
    return {
        "latent": latent,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "condition_tokens": condition_tokens,
        "condition_active_mask": condition_active_mask,
        "timestep": timestep,
        "size_scale": size_scale,
        "aspect": aspect,
        "image_coordinates": image_coords,
    }


def test_slot08_capture_dense_vs_packed_parity() -> None:
    # PackedDiT is locked to the production FA4 config (d=2560, 20Q/5KV,
    # head_dim=128, bfloat16), so a small parity model does not exist: the
    # parity check uses the full production model.  The same weights and
    # inputs must yield the same slot_08 capture through the dense (SDPA) and
    # packed (FA4 varlen) backends.
    dense = DenseDiT(**_production_kwargs()).to(device=DEVICE)  # pyright: ignore[reportArgumentType]
    dense.eval()
    packed = PackedDiT(**_production_kwargs()).to(device=DEVICE)  # pyright: ignore[reportArgumentType]
    # Identical weights for both backends (isomorphic state dicts).
    packed.load_state_dict(dense.state_dict())
    packed.eval()

    height, width = 4, 4
    inputs = _production_forward_inputs(height, width)
    image_coords = inputs["image_coordinates"]
    dense_predictions, dense_capture = dense.forward_tapped(
        inputs["latent"],
        inputs["text_tokens"],
        inputs["text_mask"],
        inputs["condition_tokens"],
        inputs["condition_active_mask"],
        inputs["timestep"],
        inputs["size_scale"],
        inputs["aspect"],
        image_coordinates=image_coords,
        growth_alpha=1.0,
        tap_slot_id=TAP_SLOT,
    )

    packed_seqs = packed.prepare_packed_sequences(
        tuple(inputs["latent"].unbind(0)),
        inputs["text_tokens"],
        inputs["text_mask"],
        (TEXT_LEN,) * 1,
        inputs["condition_tokens"],
    )
    packed_predictions, packed_capture = packed.forward_packed_tapped(
        packed_seqs,
        inputs["condition_tokens"],
        inputs["condition_active_mask"],
        inputs["timestep"],
        inputs["size_scale"],
        inputs["aspect"],
        image_coordinates=tuple(image_coords.unbind(0)),
        growth_alpha=1.0,
        tap_slot_id=TAP_SLOT,
    )

    image_tokens = height * width
    packed_capture = packed_capture.reshape(1, image_tokens, 2560)
    # Same weights + inputs: the capture must agree across backends within
    # bfloat16 rounding (different attention kernels, not different math).
    torch.testing.assert_close(
        dense_capture.float(), packed_capture.float(), rtol=1e-1, atol=1e-1
    )
    for dense_pred, packed_pred in zip(
        dense_predictions, packed_predictions, strict=True
    ):
        torch.testing.assert_close(
            dense_pred.float(), packed_pred.float(), rtol=1e-1, atol=1e-1
        )


def test_slot08_capture_stable_across_activation_checkpointing() -> None:
    # Activation checkpointing recomputes the exact same block math, so the
    # capture must be bit-stable across all three modes on one model.
    dit = _small_dit(20, linear_dtype=torch.float32)
    dit.eval()
    inputs = _forward_inputs(4, 4, dtype=torch.float32)
    reference: torch.Tensor | None = None
    for ckpt_mode in ("none", "alternating", "all"):
        dit.set_activation_checkpoint_mode(ckpt_mode)
        capture = _tapped_capture(dit, inputs)
        captured = capture.detach().float().cpu()
        torch.cuda.synchronize()  # pyright: ignore[reportUnknownMemberType]
        if reference is None:
            reference = captured.clone()
        else:
            torch.testing.assert_close(captured, reference, rtol=0.0, atol=0.0)
    assert reference is not None


def test_slot08_capture_stable_across_torch_compile() -> None:
    # The capture read is an eager outer-loop point (never a computation
    # inside a compiled block), so compiling forward_tapped must not perturb
    # the capture beyond fp32 rounding, on the same model and inputs.
    dit = _small_dit(20, linear_dtype=torch.float32)
    dit.eval()
    inputs = _forward_inputs(4, 4, dtype=torch.float32)
    eager_capture = _tapped_capture(dit, inputs).detach().float().cpu()
    torch.cuda.synchronize()  # pyright: ignore[reportUnknownMemberType]

    compiled_forward = torch.compile(dit.forward_tapped)
    _compiled_predictions, compiled_capture = compiled_forward(
        inputs["latent"],
        inputs["text_tokens"],
        inputs["text_mask"],
        inputs["condition_tokens"],
        inputs["condition_active_mask"],
        inputs["timestep"],
        inputs["size_scale"],
        inputs["aspect"],
        image_coordinates=inputs["image_coordinates"],
        growth_alpha=1.0,
        tap_slot_id=TAP_SLOT,
    )
    compiled_cpu = compiled_capture.detach().float().cpu()
    torch.cuda.synchronize()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(compiled_cpu, eager_capture, rtol=1e-4, atol=1e-4)


@pytest.fixture(scope="module")
def teacher() -> FrozenPESpatialEncoder:
    # The consuming tests all carry ``requires_teacher``; when the asset is
    # absent they skip before this fixture is ever requested.
    return FrozenPESpatialEncoder.load_asset(REPOSITORY_ROOT, TEACHER_DIR, device=DEVICE)


def _teacher_objective(
    teacher: FrozenPESpatialEncoder,
    *,
    height: int,
    width: int,
    rgb_seed: int,
) -> tuple[IRepaAlignment, torch.Tensor, torch.Tensor]:
    dit = _small_dit(20, linear_dtype=torch.bfloat16)
    dit.train()
    projector = IRepaAlignment(HIDDEN).to(device=DEVICE)
    inputs = _forward_inputs(height, width, dtype=torch.bfloat16)
    rgb = (
        2.0
        * torch.randn(
            1,
            3,
            16 * height,
            16 * width,
            generator=_seeded(rgb_seed),
            device=DEVICE,
        )
        - 1.0
    ).to(torch.bfloat16)
    with torch.no_grad():
        teacher_output = prepare_teacher_targets(teacher, rgb)
    assert tuple(teacher_output.patch_features.shape) == (1, height * width, 768)

    capture = _tapped_capture(dit, inputs)
    assert capture.dtype is torch.bfloat16
    projected = projector(capture, (height, width))
    assert tuple(projected.shape) == (1, height * width, 768)
    target = spatial_zscore_target(teacher_output.patch_features, gamma=1.0, eps=1e-5)
    assert target.dtype is torch.float32
    alignment = irepa_alignment_loss(projected, target)
    assert tuple(alignment.per_sample.shape) == (1,)
    return projector, alignment.per_sample, alignment.cosine_per_sample


@requires_teacher
def test_teacher_objective_chain_backpropagates_projector_grads(
    teacher: FrozenPESpatialEncoder
) -> None:
    projector, per_sample, cosine_per_sample = _teacher_objective(
        teacher, height=4, width=4, rgb_seed=3
    )
    lambda_weight = 0.5
    objective = (lambda_weight * per_sample).sum()
    assert bool(torch.isfinite(objective))
    objective.backward()

    weight_grad = projector.projector.weight.grad
    bias_grad = projector.projector.bias.grad
    assert weight_grad is not None and bias_grad is not None
    assert float(weight_grad.abs().max()) > 0.0
    assert float(bias_grad.abs().max()) > 0.0
    assert bool(torch.isfinite(weight_grad).all())
    assert bool(torch.isfinite(bias_grad).all())
    cosine = float(cosine_per_sample[0])
    assert -1.0 <= cosine <= 1.0


@requires_teacher
def test_lambda_zero_is_exact_zero_contribution(
    teacher: FrozenPESpatialEncoder
) -> None:
    projector, per_sample, _cosine = _teacher_objective(
        teacher, height=4, width=4, rgb_seed=11
    )
    # lambda = 0: the weighted per-sample term is an exact zero tensor even
    # though the iREPA graph is present (no skip).
    assert torch.equal(0.0 * per_sample, torch.zeros_like(per_sample))
    objective = (0.0 * per_sample).sum()
    objective.backward()
    # The projector still receives a gradient: present, but an exact zero.
    weight_grad = projector.projector.weight.grad
    assert weight_grad is not None
    assert bool((weight_grad == 0).all())
