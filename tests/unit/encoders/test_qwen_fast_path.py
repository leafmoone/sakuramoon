from __future__ import annotations

import pytest

# The DTK transformers fork (and therefore the qwen encoder module) only
# exists in the HCU training environment; skip on machines without it.
pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5",
    reason="the DTK transformers fork is not installed in this environment",
)

import math

import torch
from torch import nn

import sakuramoon.encoders.qwen as qwen_module
from sakuramoon.encoders.qwen import (
    QWEN_DENSE_LENGTHS,
    FrozenQwenEncoder,
    QwenEncoderOutput,
    probe_qwen_fast_path,
    require_qwen_fast_path,
)


class _FakeLinearAttn(nn.Module):
    def __init__(self, conv_fn: object) -> None:
        super().__init__()
        self.causal_conv1d_fn = conv_fn


class _FakeLayer(nn.Module):
    def __init__(self, conv_fn: object, block_type: str = "linear_attention") -> None:
        super().__init__()
        self.block_type = block_type
        if block_type == "linear_attention":
            self.linear_attn = _FakeLinearAttn(conv_fn)


class _FakeModel(nn.Module):
    """A 24-layer model whose layout mirrors the real hybrid Qwen3.5.

    ``layout`` is a 24-tuple of ``"linear_attention"`` / ``"full_attention"``.
    ``conv_functions`` is the matching 24-tuple of conv kernels (only the
    linear positions are inspected; full positions are ignored).
    """

    def __init__(
        self,
        conv_functions: list[object] | object,
        layout: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if layout is None:
            layout = _HYBRID_LAYOUT
        if not isinstance(conv_functions, list):
            conv_functions = [conv_functions] * 24
        self.layers = nn.ModuleList(
            _FakeLayer(conv, block_type)
            for conv, block_type in zip(conv_functions, layout)
        )


class _BareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()


# The real Qwen3_5_2B layout: 18 linear-attention layers + 6 full-attention
# layers (three linear then one full, repeated six times).  A true 24-tuple;
# the string form "linear_attention"*3 would concatenate characters.
_ONE_CYCLE = ("linear_attention",) * 3 + ("full_attention",)
_HYBRID_LAYOUT: tuple[str, ...] = _ONE_CYCLE * 6
_REAL_CONVS = [math.sin if t == "linear_attention" else None for t in _HYBRID_LAYOUT]


def test_gate_passes_when_fla_kernels_are_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_module, "is_fast_path_available", True)
    encoder = FrozenQwenEncoder(_FakeModel(_REAL_CONVS, _HYBRID_LAYOUT))

    facts = require_qwen_fast_path(encoder)

    assert facts.flag_available is True
    assert facts.conv_module == "math"
    assert facts.wired_conv_layers == 18
    assert facts.conv_layer_count == 18
    assert facts.layer_count == 24


def test_gate_fails_when_fast_path_flag_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_module, "is_fast_path_available", False)
    encoder = FrozenQwenEncoder(_FakeModel(_REAL_CONVS, _HYBRID_LAYOUT))

    with pytest.raises(RuntimeError, match="fast path is not fully active"):
        require_qwen_fast_path(encoder)


def test_gate_fails_when_a_layer_has_no_conv_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_module, "is_fast_path_available", True)
    broken = list(_REAL_CONVS)
    broken[0] = None  # drop the kernel from the first linear layer
    encoder = FrozenQwenEncoder(_FakeModel(broken, _HYBRID_LAYOUT))

    with pytest.raises(RuntimeError, match="conv kernels wired on 17/18"):
        require_qwen_fast_path(encoder)


def test_gate_fails_when_no_linear_layers_carry_conv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A model with no linear-attention layers cannot exercise the fast path.
    monkeypatch.setattr(qwen_module, "is_fast_path_available", True)
    layout = ("full_attention",) * 24
    encoder = FrozenQwenEncoder(_FakeModel([None] * 24, layout))

    with pytest.raises(RuntimeError, match="conv kernels wired on 0/0"):
        require_qwen_fast_path(encoder)


def test_gate_fails_when_model_exposes_no_decoder_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_module, "is_fast_path_available", True)
    encoder = FrozenQwenEncoder(_BareModel())

    with pytest.raises(RuntimeError, match="cannot be verified"):
        require_qwen_fast_path(encoder)


def test_probe_rejects_invalid_arguments() -> None:
    # Validation fires before any encoder/device access, so a fake-backed
    # encoder is enough and keeps the call type-clean.
    encoder = FrozenQwenEncoder(_FakeModel(math.sin))
    probe = probe_qwen_fast_path
    with pytest.raises(ValueError, match="probe_rows"):
        probe(encoder, 0)
    with pytest.raises(ValueError, match="timed_launches"):
        probe(encoder, 4, timed_launches=0)
    with pytest.raises(ValueError, match="probe_length"):
        probe(encoder, 4, probe_length=QWEN_DENSE_LENGTHS[0] - 1)


def test_probe_default_length_is_a_supported_dense_bucket() -> None:
    assert qwen_module.QWEN_FAST_PATH_PROBE_LENGTH in QWEN_DENSE_LENGTHS


def test_probe_uses_the_single_launch_dense_path() -> None:
    """The probe must follow the homogeneous path used by length-aware batches."""

    class _RecordingEncoder(FrozenQwenEncoder):
        def __init__(self) -> None:
            super().__init__(nn.Linear(2, 2))
            self.calls: list[dict[str, object]] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            *,
            dense_lengths: tuple[int, ...] | None = None,
            dense_group_size: int = qwen_module.QWEN_DENSE_GROUP_SIZE,
        ) -> QwenEncoderOutput:
            assert dense_lengths is not None
            self.calls.append({"dense_lengths": dense_lengths})
            return QwenEncoderOutput(
                hidden_states=input_ids.new_zeros(
                    (input_ids.shape[0], 1), dtype=torch.float32
                ),
                attention_mask=attention_mask,
            )

    encoder = _RecordingEncoder()
    seconds = probe_qwen_fast_path(
        encoder,
        4,
        probe_length=qwen_module.QWEN_FAST_PATH_PROBE_LENGTH,
        timed_launches=2,
    )

    assert seconds >= 0.0
    assert len(encoder.calls) == 3  # one warm-up plus two timed launches
    expected = (qwen_module.QWEN_FAST_PATH_PROBE_LENGTH,) * 4
    assert all(call["dense_lengths"] == expected for call in encoder.calls)
