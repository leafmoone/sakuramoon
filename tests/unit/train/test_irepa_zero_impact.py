from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.irepa import IRepaAlignment
from sakuramoon.objective import flow as flow_module
from sakuramoon.objective.irepa import irepa_alignment_loss
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    TrainableComposite,
    TrainableCompositeInputs,
    TrainableCompositeIRepaOutput,
)

HIDDEN = 16
BATCH = 2
INPUT_CHANNELS = 8
GRID_H = 4
GRID_W = 4
IMAGE_TOKENS = GRID_H * GRID_W
SEQ_LEN = 10
MAIN_TOKENS = 6
COND_TOKENS = 8
TAP_SLOT = 8

# Depth-20 active order; slot 8 sits at index 6 (the capture point).  Every
# block after it is strictly downstream of the capture and must receive no
# iREPA gradient.
POST_CAPTURE_BLOCKS = (9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)
PRE_CAPTURE_BLOCKS = (0, 1, 3, 4, 6, 7, 8)


class _SgdAdapter:
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(parameters, lr=0.1)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _small_dit() -> DenseDiT:
    return DenseDiT(
        depth=20,
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN,
        intermediate_size=32,
        q_heads=2,
        kv_heads=1,
        head_dim=8,
        rope_nope_dim=0,
        rope_y_dim=4,
        rope_x_dim=4,
        rope_position_scale=1.0,
        rope_theta=10.0,
        norm_eps=1e-6,
        timestep_dim=256,
        size_dim=64,
        aspect_dim=64,
        condition_hidden_size=1024,
        stable_slot_count=24,
        modulation_chunks=6,
        final_modulation_size=32,
        out_channels=INPUT_CHANNELS,
        condition_token_count=COND_TOKENS,
        modality_init_std=0.02,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    )


def _small_text() -> TextConditioner:
    return TextConditioner(
        input_size=INPUT_CHANNELS,
        adapter_size=HIDDEN,
        output_size=HIDDEN,
        groups=2,
        attention_heads=2,
        norm_eps=1e-6,
        mix_gate_init=0.0,
        layer_scale_init=1.0,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    )


def _small_condition_tokens() -> ConditionTokenEncoder:
    return ConditionTokenEncoder(
        input_size=INPUT_CHANNELS,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        output_size=HIDDEN,
        token_count=COND_TOKENS,
        attention_heads=2,
        norm_eps=1e-6,
        init_std=0.02,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    )


def _inputs(seed: int) -> TrainableCompositeInputs:
    generator = torch.Generator().manual_seed(seed)
    # production Qwen states arrive as bf16 (the frozen encoder's dtype)
    qwen_states = torch.randn(
        BATCH, SEQ_LEN, 7, INPUT_CHANNELS, generator=generator
    ).bfloat16()
    # one [H*W, 2] per-sample token coordinate map (row-major raster order)
    coordinates = image_coordinates(GRID_H, GRID_W, device=torch.device("cpu"))
    return TrainableCompositeInputs(
        qwen_states=qwen_states,
        main_token_indices=torch.randint(
            0, SEQ_LEN, (BATCH, MAIN_TOKENS), generator=generator
        ),
        main_mask=torch.ones(BATCH, MAIN_TOKENS, dtype=torch.bool),
        main_token_lengths=(MAIN_TOKENS,) * BATCH,
        condition_token_indices=torch.randint(
            0, SEQ_LEN, (BATCH, COND_TOKENS), generator=generator
        ),
        condition_mask=torch.ones(BATCH, COND_TOKENS, dtype=torch.bool),
        use_null_condition=torch.zeros(BATCH, dtype=torch.bool),
        active_condition_sample_indices=torch.arange(BATCH),
        latents=tuple(
            torch.randn(INPUT_CHANNELS, GRID_H, GRID_W, generator=generator)
            .bfloat16()
            for _ in range(BATCH)
        ),
        image_coordinates=tuple(
            coordinates.unsqueeze(0).repeat(BATCH, 1, 1).unbind(0)
        ),
        timestep=torch.tensor([0.25, 0.75], dtype=torch.float32),
        size_scale=torch.tensor([1.0, 1.0], dtype=torch.float32),
        aspect=torch.zeros(BATCH, dtype=torch.float32),
        growth_alpha=1.0,
    )


def _targets(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        BATCH, IMAGE_TOKENS, 768, generator=generator
    ).detach()


def _open_gates(module: TrainableComposite) -> None:
    """Perturb the zero-initialized modulation so blocks stop being identity.

    A freshly built DiT has all-zero modulation (gates and shifts from the
    zero-initialized conditioner projections and biases), which makes every
    block an exact identity map: an iREPA-only backward can then only reach
    the gate parameters and the projector, never the block weights.  Opening
    the gates restores the general-case routing this test targets.
    """

    with torch.no_grad():
        conditioner = module.dit.conditioner
        for bias in conditioner.block_biases.values():
            bias.normal_(std=0.05)
        conditioner.shared_block_projection.weight.normal_(std=0.05)
        module.text.mix_gate.fill_(0.1)


def _main_loss(predictions: tuple[torch.Tensor, ...], seed: int) -> torch.Tensor:
    # seeded generator (not the global RNG) so the A/B runs draw identical
    # state/clean tensors for bit-exact comparison
    generator = torch.Generator().manual_seed(seed)
    prediction_batch = torch.stack(predictions)
    state = torch.randn(
        prediction_batch.shape,
        generator=generator,
        dtype=prediction_batch.dtype,
        device=prediction_batch.device,
    )
    clean = torch.randn(
        prediction_batch.shape,
        generator=generator,
        dtype=prediction_batch.dtype,
        device=prediction_batch.device,
    )
    result = flow_module.flow_matching_loss(
        prediction_batch,
        state,
        clean,
        torch.tensor([0.25, 0.75], dtype=torch.float32),
        t_eps=flow_module._T_EPS,
        noise_observation_boundary=flow_module._NOISE_OBSERVATION_BOUNDARY,
    )
    return result.per_sample


def _build_pair() -> tuple[TrainableComposite, TrainableComposite, TrainableCompositeInputs]:
    torch.manual_seed(4242)  # pyright: ignore[reportUnknownMemberType]
    dit_a = _small_dit()
    text_a = _small_text()
    condition_a = _small_condition_tokens()
    composite_a = TrainableComposite(
        dit=dit_a,
        text=text_a,
        condition_tokens=condition_a,
    )
    composite_b = TrainableComposite(
        dit=_small_dit(),
        text=_small_text(),
        condition_tokens=_small_condition_tokens(),
        irepa_alignment=IRepaAlignment(HIDDEN),
    )
    composite_b.dit.load_state_dict(dit_a.state_dict())
    composite_b.text.load_state_dict(text_a.state_dict())
    composite_b.condition_tokens.load_state_dict(condition_a.state_dict())
    composite_b.bind_irepa_tap_slot(TAP_SLOT)
    inputs = _inputs(seed=7)
    return composite_a, composite_b, inputs


def _legacy_grad_snapshot(module: TrainableComposite) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        if name.startswith("irepa_alignment."):
            continue
        assert parameter.grad is not None
        grads[name] = parameter.grad.detach().clone()
    return grads


def test_enabled_lambda_zero_is_bit_identical_to_disabled() -> None:
    composite_a, composite_b, inputs = _build_pair()

    with torch.no_grad():
        legacy_a = {
            name: tensor.clone()
            for name, tensor in composite_a.state_dict().items()
        }
        legacy_b = {
            name: tensor.clone()
            for name, tensor in composite_b.state_dict().items()
            if name in legacy_a
        }
        for name, tensor_a in legacy_a.items():
            assert torch.equal(tensor_a, legacy_b[name])

    out_a = composite_a(inputs)
    out_b = composite_b(inputs)

    # legacy contract: a plain tuple when iREPA is absent
    assert type(out_a) is tuple
    assert isinstance(out_b, TrainableCompositeIRepaOutput)
    for prediction_a, prediction_b in zip(out_a, out_b.predictions, strict=True):
        assert torch.equal(prediction_a, prediction_b)
        assert prediction_a.shape == (INPUT_CHANNELS, GRID_H, GRID_W)

    main_a = _main_loss(out_a, seed=11)
    main_b = _main_loss(out_b.predictions, seed=11)
    assert torch.equal(main_a, main_b)

    step_a = SingleGpuStep(
        composite_a,
        _SgdAdapter(list(composite_a.parameters())),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )
    step_a.backward(main_a)

    # Phase-4 arithmetic (mirrors runtime._loss): the iREPA term stays in
    # the graph at lambda=0 (spec: no skip) as an exact zero.
    targets = _targets(seed=13)
    alignment = irepa_alignment_loss(out_b.projected_student_features, targets)
    assert out_b.projected_student_features.shape == (BATCH, IMAGE_TOKENS, 768)
    per_sample_b = main_b + 0.0 * alignment.per_sample

    step_b = SingleGpuStep(
        composite_b,
        _SgdAdapter(list(composite_b.parameters())),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
        irepa_projector=True,
    )
    step_b.backward(per_sample_b)

    # every legacy gradient is bit-identical to the disabled run
    grads_a = _legacy_grad_snapshot(composite_a)
    grads_b = _legacy_grad_snapshot(composite_b)
    assert set(grads_a) == set(grads_b)
    for name in sorted(grads_a):
        assert torch.equal(grads_a[name], grads_b[name]), name

    # the projector ran (graph reachable) but received an exact zero grad
    projector_grads = {
        name: parameter.grad
        for name, parameter in composite_b.named_parameters()
        if name.startswith("irepa_alignment.")
    }
    assert set(projector_grads) == {
        "irepa_alignment.projector.weight",
        "irepa_alignment.projector.bias",
    }
    for name, grad in projector_grads.items():
        assert grad is not None
        assert torch.equal(grad, torch.zeros_like(grad)), name

    result_a = step_a.finish_update()
    result_b = step_b.finish_update()
    assert result_a.irepa_projector_grad_norm == 0.0
    assert result_b.irepa_projector_grad_norm == 0.0
    # grad state is fully restored after the update
    assert all(
        parameter.grad is None for parameter in composite_b.parameters()
    )


def test_lambda_positive_routes_irepa_grads_only_to_pre_capture_parameters() -> None:
    _composite_a, composite_b, inputs = _build_pair()
    _open_gates(composite_b)
    targets = _targets(seed=13)

    # run 1: iREPA-only backward.  Isolating the alignment graph makes
    # presence/absence of each parameter's gradient an EXACT check
    # (immune to bf16 grad quantization of any main gradient).
    out_1 = composite_b(inputs)
    alignment_1 = irepa_alignment_loss(out_1.projected_student_features, targets)
    step_irepa_only = SingleGpuStep(
        composite_b,
        _SgdAdapter(list(composite_b.parameters())),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
        irepa_projector=True,
    )
    step_irepa_only.backward(alignment_1.per_sample)
    irepa_only_grads: dict[str, torch.Tensor | None] = {
        name: parameter.grad
        for name, parameter in composite_b.named_parameters()
    }
    step_irepa_only.finish_update()

    def _has_irepa_grad(prefix: str) -> bool:
        return any(
            grad is not None and bool((grad != 0.0).any())
            for name, grad in irepa_only_grads.items()
            if name.startswith(prefix)
        )

    # the projector and the tapped slot's own block (capture is AFTER it)
    assert _has_irepa_grad("irepa_alignment.")
    assert _has_irepa_grad(f"dit.blocks.slot_{TAP_SLOT:02d}.")
    # upstream conditioning reaches the capture through the joint graph
    assert _has_irepa_grad("text.")
    assert _has_irepa_grad("condition_tokens.")
    # strictly downstream of the capture: an EXACT zero iREPA gradient
    for name, grad in irepa_only_grads.items():
        if any(
            name.startswith(f"dit.blocks.slot_{slot:02d}.")
            for slot in POST_CAPTURE_BLOCKS
        ) or name.startswith("dit.output_head."):
            assert grad is None or bool((grad == 0.0).all()), name

    # run 2: combined main + lambda*irepa update (a fresh forward graph)
    out_2 = composite_b(inputs)
    main_2 = _main_loss(out_2.predictions, seed=11)
    alignment_2 = irepa_alignment_loss(out_2.projected_student_features, targets)
    step_enabled = SingleGpuStep(
        composite_b,
        _SgdAdapter(list(composite_b.parameters())),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
        irepa_projector=True,
    )
    step_enabled.backward(main_2 + 0.5 * alignment_2.per_sample)
    result = step_enabled.finish_update()
    assert result.irepa_projector_grad_norm > 0.0


def test_projector_input_dtype_contract_fails_closed() -> None:
    # the projector requires the bf16 hidden state; a float32 capture is a
    # contract violation, not a silent upcast
    _composite_a, composite_b, inputs = _build_pair()
    assert composite_b.irepa_alignment is not None
    assert composite_b.irepa_tap_slot_id == TAP_SLOT
    with torch.no_grad():
        with pytest.raises(TypeError, match="bfloat16"):
            composite_b._project_student_capture(
                inputs,
                torch.randn(BATCH, IMAGE_TOKENS, HIDDEN),
            )
        projected = composite_b._project_student_capture(
            inputs,
            torch.randn(BATCH, IMAGE_TOKENS, HIDDEN).bfloat16(),
        )
    assert projected.dtype is torch.bfloat16
    assert projected.shape == (BATCH, IMAGE_TOKENS, 768)


def test_flow_main_loss_is_fp32_per_sample_vector() -> None:
    # the step contract requires an FP32 [B] main loss; the flow objective
    # produces it from bf16 predictions (the .float() cast in flow.py)
    composite_a, _composite_b, inputs = _build_pair()
    out_a = composite_a(inputs)
    main = _main_loss(out_a, seed=11)
    assert main.dtype is torch.float32
    assert main.shape == (BATCH,)
    assert bool(torch.isfinite(main).all())
