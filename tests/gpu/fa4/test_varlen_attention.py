from __future__ import annotations

import importlib.metadata
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

import sakuramoon.model.attention as attention_module
from sakuramoon.model.attention import (
    FA4_PACK_GQA,
    AcceptedCuSeqlens,
    DenseGQAAttention,
    FA4VarlenGQAAttention,
    ValidatedCuSeqlens,
    accept_fa4_boundaries,
    accepted_sample_indices,
    build_validated_cu_seqlens,
    dense_attention_mask,
    fa4_varlen_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA-compatible HCU is required"
)

BUCKETS_512 = (
    (1024, 256),
    (896, 288),
    (832, 320),
    (736, 352),
    (672, 384),
    (640, 416),
    (576, 448),
    (544, 480),
    (512, 512),
    (480, 544),
    (448, 576),
    (416, 640),
    (384, 672),
    (352, 736),
    (320, 832),
    (288, 896),
    (256, 1024),
)


def _dense_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: tuple[int, ...],
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start, end in pairwise(offsets):
        query_i = query[start:end].transpose(0, 1).unsqueeze(0)
        key_i = key[start:end].transpose(0, 1).unsqueeze(0)
        value_i = value[start:end].transpose(0, 1).unsqueeze(0)
        length = end - start
        mask = torch.ones(1, 1, length, length, device="cuda", dtype=torch.bool)
        output_i = F.scaled_dot_product_attention(
            query_i,
            key_i,
            value_i,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        outputs.append(output_i.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


def _production_attention() -> FA4VarlenGQAAttention:
    return FA4VarlenGQAAttention(
        hidden_size=2560,
        q_heads=20,
        kv_heads=5,
        head_dim=128,
        rope_nope_dim=32,
        rope_y_dim=48,
        rope_x_dim=48,
        rope_position_scale=16.0,
        rope_theta=1000.0,
        norm_eps=1e-6,
        linear_dtype=torch.bfloat16,
        projection_bias=False,
        dropout=0.0,
    ).cuda()


def _dense_production_attention() -> DenseGQAAttention:
    return DenseGQAAttention(
        hidden_size=2560,
        q_heads=20,
        kv_heads=5,
        head_dim=128,
        rope_nope_dim=32,
        rope_y_dim=48,
        rope_x_dim=48,
        rope_position_scale=16.0,
        rope_theta=1000.0,
        norm_eps=1e-6,
        linear_dtype=torch.bfloat16,
        projection_bias=False,
        dropout=0.0,
    ).cuda()


def _validated_boundaries(
    offsets: tuple[int, ...],
) -> AcceptedCuSeqlens:
    lengths = tuple(end - start for start, end in pairwise(offsets))
    boundaries = build_validated_cu_seqlens(
        lengths,
        device=torch.device("cuda"),
    )
    return accept_fa4_boundaries(
        boundaries,
        total_tokens=offsets[-1],
        batch_size=len(lengths),
        device=torch.device("cuda"),
    )


def test_locked_das_fa2_varlen_matches_dense_and_isolates_samples() -> None:
    assert (
        importlib.metadata.version("flash-attn")
        == "2.8.3+das.opt1.dtk2604.torch290"
    )
    torch.manual_seed(123)  # pyright: ignore[reportUnknownMemberType]
    lengths = (113, 197)
    offsets = (0, lengths[0], sum(lengths))
    boundaries = _validated_boundaries(offsets)
    total = offsets[-1]
    query_data = torch.randn(total, 20, 128, device="cuda", dtype=torch.bfloat16)
    key_data = torch.randn(total, 5, 128, device="cuda", dtype=torch.bfloat16)
    value_data = torch.randn(total, 5, 128, device="cuda", dtype=torch.bfloat16)
    loss_weight = torch.randn(total, 20, 128, device="cuda", dtype=torch.float32)

    query = query_data.clone().requires_grad_()
    key = key_data.clone().requires_grad_()
    value = value_data.clone().requires_grad_()
    output = fa4_varlen_attention(query, key, value, boundaries)
    loss = (output.float() * loss_weight).mean()
    loss.backward()  # pyright: ignore[reportUnknownMemberType]

    dense_query = query_data.clone().requires_grad_()
    dense_key = key_data.clone().requires_grad_()
    dense_value = value_data.clone().requires_grad_()
    dense_output = _dense_reference(
        dense_query,
        dense_key,
        dense_value,
        offsets,
    )
    dense_loss = (dense_output.float() * loss_weight).mean()
    dense_loss.backward()  # pyright: ignore[reportUnknownMemberType]

    assert output.shape == (total, 20, 128)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    query_grad = query.grad
    key_grad = key.grad
    value_grad = value.grad
    dense_query_grad = dense_query.grad
    dense_key_grad = dense_key.grad
    dense_value_grad = dense_value.grad
    assert query_grad is not None and key_grad is not None and value_grad is not None
    assert (
        dense_query_grad is not None
        and dense_key_grad is not None
        and dense_value_grad is not None
    )
    assert (output.float() - dense_output.float()).abs().max().item() <= 0.008
    assert abs(loss.item() - dense_loss.item()) <= 5e-7
    assert (query_grad.float() - dense_query_grad.float()).abs().max().item() <= 1e-7
    assert (key_grad.float() - dense_key_grad.float()).abs().max().item() <= 3e-6
    assert (value_grad.float() - dense_value_grad.float()).abs().max().item() <= 3e-6

    learning_rate = 128.0
    with torch.no_grad():
        query.add_(query_grad, alpha=-learning_rate)
        key.add_(key_grad, alpha=-learning_rate)
        value.add_(value_grad, alpha=-learning_rate)
        dense_query.add_(dense_query_grad, alpha=-learning_rate)
        dense_key.add_(dense_key_grad, alpha=-learning_rate)
        dense_value.add_(dense_value_grad, alpha=-learning_rate)
    assert not torch.equal(query, query_data)
    assert not torch.equal(key, key_data)
    assert not torch.equal(value, value_data)
    torch.testing.assert_close(query, dense_query, atol=0.008, rtol=0)
    torch.testing.assert_close(key, dense_key, atol=0.008, rtol=0)
    torch.testing.assert_close(value, dense_value, atol=0.008, rtol=0)

    changed_key = key_data.clone()
    changed_value = value_data.clone()
    changed_key[lengths[0] :] = torch.randn_like(changed_key[lengths[0] :]) * 100
    changed_value[lengths[0] :] = torch.randn_like(changed_value[lengths[0] :]) * 100
    changed = fa4_varlen_attention(
        query_data,
        changed_key,
        changed_value,
        boundaries,
    )
    torch.testing.assert_close(
        changed[: lengths[0]],
        output.detach()[: lengths[0]],
        atol=0,
        rtol=0,
    )


def test_all_17_image_shapes_and_text_boundaries_execute() -> None:
    torch.manual_seed(321)  # pyright: ignore[reportUnknownMemberType]
    text_lengths = (0, 39, 512)
    for height, width in BUCKETS_512:
        for text_length in text_lengths:
            length = height // 16 * (width // 16) + 4 + text_length
            boundaries = _validated_boundaries((0, length))
            query = torch.randn(length, 20, 128, device="cuda", dtype=torch.bfloat16)
            key = torch.randn(length, 5, 128, device="cuda", dtype=torch.bfloat16)
            value = torch.randn(length, 5, 128, device="cuda", dtype=torch.bfloat16)

            output = fa4_varlen_attention(query, key, value, boundaries)

            assert output.shape == query.shape
            assert torch.isfinite(output).all()


def test_full_fa4_attention_forward_backward_and_update() -> None:
    torch.manual_seed(456)  # pyright: ignore[reportUnknownMemberType]
    module = _production_attention()
    lengths = (37, 53)
    total = sum(lengths)
    tokens = torch.randn(total, 2560, device="cuda", dtype=torch.bfloat16)
    coordinates = torch.zeros(total, 2, device="cuda", dtype=torch.float32)
    boundaries = _validated_boundaries((0, lengths[0], total))
    loss_weight = torch.randn_like(tokens, dtype=torch.float32)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    weight_before = module.q_proj.weight.detach().clone()

    output = module(tokens, boundaries, coordinates)
    loss = (output.float() * loss_weight).sum() / total
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    query_grad = module.q_proj.weight.grad
    assert output.shape == tokens.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert not torch.equal(weight_before, module.q_proj.weight)


def test_sm120_production_asymmetric_length_matrix_repeats_backward() -> None:
    assert FA4_PACK_GQA is False
    torch.manual_seed(457)  # pyright: ignore[reportUnknownMemberType]
    length_pairs = (
        (299, 774),
        (774, 299),
        (319, 320),
        (383, 641),
        (511, 769),
    )

    for _ in range(4):
        for lengths in length_pairs:
            total = sum(lengths)
            boundaries = _validated_boundaries((0, lengths[0], total))
            query = torch.randn(
                total,
                20,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            key = torch.randn(
                total,
                5,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            value = torch.randn(
                total,
                5,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )

            output = fa4_varlen_attention(query, key, value, boundaries)
            torch.cuda.synchronize()
            output.float().square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
            torch.cuda.synchronize()

            assert query.grad is not None and torch.isfinite(query.grad).all()
            assert key.grad is not None and torch.isfinite(key.grad).all()
            assert value.grad is not None and torch.isfinite(value.grad).all()


@pytest.mark.parametrize("lengths", [(), (0,), (-1,), (True,)])
def test_boundary_factory_rejects_invalid_host_lengths(
    lengths: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        build_validated_cu_seqlens(
            lengths,
            device=torch.device("cuda"),
        )


def _forge_boundaries(
    tensor: torch.Tensor,
    *,
    sequence_lengths: tuple[int, ...] = (2,),
    total_tokens: int = 2,
    max_seqlen: int = 2,
    batch_size: int = 1,
) -> ValidatedCuSeqlens:
    boundaries = object.__new__(ValidatedCuSeqlens)
    object.__setattr__(boundaries, "tensor", tensor)
    object.__setattr__(boundaries, "sequence_lengths", sequence_lengths)
    object.__setattr__(boundaries, "total_tokens", total_tokens)
    object.__setattr__(boundaries, "max_seqlen", max_seqlen)
    object.__setattr__(boundaries, "batch_size", batch_size)
    return boundaries


@pytest.mark.parametrize(
    "case",
    ["dtype", "shape", "contiguous", "host_metadata", "offset_values"],
)
def test_forged_boundary_handle_fails_before_native_kernel(case: str) -> None:
    if case == "dtype":
        boundaries = _forge_boundaries(torch.tensor([0.0, 2.0], device="cuda"))
    elif case == "shape":
        boundaries = _forge_boundaries(
            torch.tensor([0], dtype=torch.int32, device="cuda")
        )
    elif case == "contiguous":
        boundaries = _forge_boundaries(
            torch.tensor([0, 99, 2], dtype=torch.int32, device="cuda")[::2]
        )
    elif case == "host_metadata":
        boundaries = _forge_boundaries(
            torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            sequence_lengths=(1, 1),
        )
    else:
        boundaries = _forge_boundaries(
            torch.tensor([0, 3, 4], dtype=torch.int32, device="cuda"),
            sequence_lengths=(2, 2),
            total_tokens=4,
            max_seqlen=2,
            batch_size=2,
        )

    with pytest.raises(ValueError, match="metadata|values"):
        accept_fa4_boundaries(
            boundaries,
            total_tokens=boundaries.total_tokens,
            batch_size=boundaries.batch_size,
            device=torch.device("cuda"),
        )


def test_post_construction_boundary_mutation_fails_at_packed_entry() -> None:
    boundaries = build_validated_cu_seqlens(
        (2, 2),
        device=torch.device("cuda"),
    )
    boundaries.tensor[1] = 3

    with pytest.raises(ValueError, match="differ from validated host lengths"):
        accept_fa4_boundaries(
            boundaries,
            total_tokens=4,
            batch_size=2,
            device=torch.device("cuda"),
        )


def test_unaccepted_public_boundaries_cannot_reach_native_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.zeros(2, 20, 128, dtype=torch.bfloat16, device="cuda")
    key = torch.zeros(2, 5, 128, dtype=torch.bfloat16, device="cuda")
    boundaries = build_validated_cu_seqlens((2,), device=torch.device("cuda"))
    native_calls: list[object] = []

    def record_native_call(*args: object, **kwargs: object) -> object:
        native_calls.append((args, kwargs))
        raise AssertionError("native kernel must not be called")

    monkeypatch.setattr(
        attention_module,
        "_flash_attn_varlen_func",
        record_native_call,
    )

    with pytest.raises(TypeError, match="accepted at the packed entry"):
        fa4_varlen_attention(query, key, torch.zeros_like(key), boundaries)  # pyright: ignore[reportArgumentType]
    assert native_calls == []


def test_sample_routing_uses_the_accepted_host_identity() -> None:
    boundaries = _validated_boundaries((0, 2, 4))

    assert torch.equal(
        accepted_sample_indices(boundaries),
        torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64),
    )


def _pack_valid_tokens(padded: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.cat(
        [row[:length] for row, length in zip(padded, lengths, strict=True)]
    )


def _p99_absolute_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    errors = (actual.float() - expected.float()).abs().flatten()
    return torch.quantile(errors, 0.99).item()


def test_full_fa4_module_matches_dense_reference_and_repeat_control() -> None:
    torch.manual_seed(789)  # pyright: ignore[reportUnknownMemberType]
    lengths = (11, 17)
    max_length = max(lengths)
    total = sum(lengths)
    offsets = (0, lengths[0], total)
    boundaries = _validated_boundaries(offsets)
    tokens = torch.randn(total, 2560, device="cuda", dtype=torch.bfloat16)
    coordinates = torch.randn(total, 2, device="cuda", dtype=torch.float32)
    loss_weight = torch.randn_like(tokens, dtype=torch.float32)

    padded_tokens = torch.zeros(
        len(lengths), max_length, 2560, device="cuda", dtype=torch.bfloat16
    )
    padded_coordinates = torch.zeros(
        len(lengths), max_length, 2, device="cuda", dtype=torch.float32
    )
    token_mask = torch.zeros(len(lengths), max_length, device="cuda", dtype=torch.bool)
    for index, (start, end) in enumerate(pairwise(offsets)):
        length = end - start
        padded_tokens[index, :length] = tokens[start:end]
        padded_coordinates[index, :length] = coordinates[start:end]
        token_mask[index, :length] = True

    fa4_module = _production_attention()
    dense_module = _dense_production_attention()
    dense_module.load_state_dict(fa4_module.state_dict())

    fa4_output = fa4_module(tokens, boundaries, coordinates)
    fa4_loss = (fa4_output.float() * loss_weight).mean()
    fa4_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    fa4_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in fa4_module.named_parameters()
        if parameter.grad is not None
    }

    fa4_module.zero_grad(set_to_none=True)
    repeat_output = fa4_module(tokens, boundaries, coordinates)
    repeat_loss = (repeat_output.float() * loss_weight).mean()
    repeat_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    repeat_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in fa4_module.named_parameters()
        if parameter.grad is not None
    }

    dense_output_padded = dense_module(
        padded_tokens,
        dense_attention_mask(token_mask),
        padded_coordinates,
    )
    dense_output = _pack_valid_tokens(dense_output_padded, lengths)
    dense_loss = (dense_output.float() * loss_weight).mean()
    dense_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    dense_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in dense_module.named_parameters()
        if parameter.grad is not None
    }

    expected_gradient_names = {name for name, _ in fa4_module.named_parameters()}
    assert fa4_gradients.keys() == expected_gradient_names
    assert fa4_gradients.keys() == repeat_gradients.keys() == dense_gradients.keys()
    repeat_output_p99 = _p99_absolute_error(fa4_output, repeat_output)
    output_p99_tolerance = max(0.002, 8.0 * repeat_output_p99)
    assert _p99_absolute_error(fa4_output, dense_output) <= output_p99_tolerance
    assert abs(fa4_loss.item() - dense_loss.item()) <= max(
        2e-6, 8.0 * abs(fa4_loss.item() - repeat_loss.item())
    )
    assert (fa4_output.float() - dense_output.float()).abs().max().item() <= 0.016

    for name, fa4_gradient in fa4_gradients.items():
        repeat_p99 = _p99_absolute_error(fa4_gradient, repeat_gradients[name])
        reference_p99 = _p99_absolute_error(fa4_gradient, dense_gradients[name])
        gradient_scale_p99 = torch.quantile(
            dense_gradients[name].float().abs().flatten(), 0.99
        ).item()
        gradient_tolerance = max(
            5e-5,
            8.0 * repeat_p99,
            1.25 * gradient_scale_p99,
        )
        assert reference_p99 <= gradient_tolerance, name

    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in fa4_module.named_parameters()
    }
    learning_rate = 64.0
    fa4_optimizer = torch.optim.SGD(fa4_module.parameters(), lr=learning_rate)
    dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=learning_rate)
    fa4_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    dense_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    for (fa4_name, fa4_parameter), (dense_name, dense_parameter) in zip(
        fa4_module.named_parameters(), dense_module.named_parameters(), strict=True
    ):
        assert fa4_name == dense_name
        update_p99 = _p99_absolute_error(fa4_parameter, dense_parameter)
        gradient_scale_p99 = torch.quantile(
            dense_gradients[fa4_name].float().abs().flatten(), 0.99
        ).item()
        update_tolerance = max(0.002, 80.0 * gradient_scale_p99)
        assert update_p99 <= update_tolerance, fa4_name
        assert not torch.equal(parameter_before[fa4_name], fa4_parameter), fa4_name
